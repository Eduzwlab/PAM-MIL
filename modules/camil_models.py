
import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple, Union


__all__ = [
    "CAMILBase",
    "CAMILClassification",
    "CAMILRegression",
    "GrazianiRegression",
    "MILModel",
    "Attention",
]


TensorOrArgs = Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Sequence[torch.Tensor],
]


def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    """
    Attention block:
        Linear(n_in, n_latent) -> Tanh -> Linear(n_latent, 1)

    If n_latent is None, use (n_in + 1) // 2.
    """
    n_latent = n_latent or (n_in + 1) // 2
    return nn.Sequential(
        nn.Linear(n_in, n_latent),
        nn.Tanh(),
        nn.Linear(n_latent, 1),
    )


class CAMILBase(nn.Module):
    """
    Shared CAMIL backbone:
      1) FC + ReLU encoder
      2) masked attention for padded bags
      3) attention-weighted bag embedding

    Supported input:
      - bags: Tensor (N, D) or (B, N, D)
      - (bags, lens):
            bags: Tensor (N, D) or (B, N, D)
            lens: Tensor (B,) or scalar
    """

    def __init__(
        self,
        n_feats: int,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        emb_dim: int = 256,
        attn_dim: Optional[int] = None,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_feats = n_feats
        self.emb_dim = emb_dim

        self.encoder = encoder or nn.Sequential(
            nn.Linear(n_feats, emb_dim),
            nn.ReLU(inplace=True),
        )

        self.attention = attention or Attention(emb_dim, attn_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self._init_linear_weights()

    def _init_linear_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _parse_input(
        self, args: TensorOrArgs
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Returns
        -------
        bags     : Tensor (B, N, D)
        lens     : Tensor (B,)
        squeezed : bool
            whether original input was single bag (N, D)
        """
        if isinstance(args, torch.Tensor):
            bags = args
            lens = None
        elif isinstance(args, (tuple, list)) and len(args) >= 1:
            bags = args[0]
            lens = args[1] if len(args) > 1 else None
        else:
            raise TypeError(
                "Input must be a Tensor, (bags, lens) tuple, or list-like sequence."
            )

        if bags.dim() == 2:
            bags = bags.unsqueeze(0)  # (1, N, D)
            squeezed = True
        elif bags.dim() == 3:
            squeezed = False
        else:
            raise ValueError(
                f"Expected bags with shape (N, D) or (B, N, D), got {tuple(bags.shape)}"
            )

        B, N, _ = bags.shape

        if lens is None:
            lens = torch.full((B,), N, dtype=torch.long, device=bags.device)
        else:
            if not isinstance(lens, torch.Tensor):
                lens = torch.as_tensor(lens, dtype=torch.long, device=bags.device)
            else:
                lens = lens.to(device=bags.device, dtype=torch.long)

            if lens.dim() == 0:
                lens = lens.unsqueeze(0)

            if lens.numel() == 1 and B > 1:
                lens = lens.expand(B)

            if lens.shape[0] != B:
                raise ValueError(
                    f"lens batch size mismatch: bags batch={B}, lens shape={tuple(lens.shape)}"
                )

        return bags, lens, squeezed

    def _masked_attention_scores(
        self, embeddings: torch.Tensor, lens: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : Tensor (B, N, E)
        lens       : Tensor (B,)

        Returns
        -------
        A_norm : Tensor (B, N, 1)
        """
        B, N, _ = embeddings.shape
        attention_scores = self.attention(self.attn_dropout(embeddings))  # (B, N, 1)

        idx = torch.arange(N, device=embeddings.device).unsqueeze(0).expand(B, N)
        attention_mask = (idx < lens.unsqueeze(1)).unsqueeze(-1)  # (B, N, 1)

        masked_attention = torch.where(
            attention_mask,
            attention_scores,
            torch.full_like(attention_scores, -1e10),
        )
        return torch.softmax(masked_attention, dim=1)

    def _encode_bag(
        self, args: TensorOrArgs
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Returns
        -------
        M        : Tensor (B, E)
            attention-weighted bag embedding
        A_out    : Tensor (B, N)
            normalized attention weights
        squeezed : bool
        """
        bags, lens, squeezed = self._parse_input(args)
        embeddings = self.encoder(bags)                     # (B, N, E)
        A = self._masked_attention_scores(embeddings, lens) # (B, N, 1)
        M = (A * embeddings).sum(dim=1)                     # (B, E)
        A_out = A.squeeze(-1)                               # (B, N)
        return M, A_out, squeezed

    def get_attention_weights(self, args: TensorOrArgs):
        with torch.no_grad():
            out = self.forward(args)
        return out


class CAMILClassification(CAMILBase):
    """
    CAMIL classification head:
        Flatten -> BatchNorm1d -> Dropout -> FC

    Notes
    -----
    - forward() returns logits, not probabilities.
    - Binary classification: use BCEWithLogitsLoss with n_out=1
    - Multi-class classification: use CrossEntropyLoss with n_out>1
    """

    def __init__(
        self,
        n_feats: int,
        n_out: int = 1,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        emb_dim: int = 256,
        attn_dim: Optional[int] = None,
        dropout: float = 0.25,
        head_dropout: float = 0.5,
        use_bn: bool = True,
    ) -> None:
        super().__init__(
            n_feats=n_feats,
            encoder=encoder,
            attention=attention,
            emb_dim=emb_dim,
            attn_dim=attn_dim,
            dropout=dropout,
        )

        head_layers = [nn.Flatten()]
        if use_bn:
            head_layers.append(nn.BatchNorm1d(emb_dim))
        head_layers.append(nn.Dropout(head_dropout))
        head_layers.append(nn.Linear(emb_dim, n_out))
        self.head = nn.Sequential(*head_layers)

        self.n_out = n_out
        self._init_linear_weights()

    def forward(self, args: TensorOrArgs):
        M, A, squeezed = self._encode_bag(args)
        logits = self.head(M)  # (B, n_out)

        if squeezed:
            return logits.squeeze(0), A.squeeze(0)
        return logits, A

    def forward_test(self, args: TensorOrArgs):
        logits, _ = self.forward(args)
        return logits

    def predict_proba(self, args: TensorOrArgs):
        logits, A = self.forward(args)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        if self.n_out == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=1)

        if squeezed:
            return probs.squeeze(0), A
        return probs, A


class CAMILRegression(CAMILBase):
    """
    CAMIL regression head:
        Flatten -> FC

    Parameters
    ----------
    n_out : int
        output dimension
        - n_out=1 : scalar regression
        - n_out=3 : multi-output regression / soft-label regression
    bounded : bool
        if True, apply sigmoid to bound outputs in [0,1]
    """

    def __init__(
        self,
        n_feats: int,
        n_out: int = 1,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        emb_dim: int = 256,
        attn_dim: Optional[int] = None,
        dropout: float = 0.25,
        bounded: bool = False,
    ) -> None:
        super().__init__(
            n_feats=n_feats,
            encoder=encoder,
            attention=attention,
            emb_dim=emb_dim,
            attn_dim=attn_dim,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(emb_dim, n_out),
        )

        self.n_out = n_out
        self.bounded = bounded
        self._init_linear_weights()

    def forward(self, args: TensorOrArgs):
        M, A, squeezed = self._encode_bag(args)
        preds = self.head(M)

        if self.bounded:
            preds = torch.sigmoid(preds)

        if squeezed:
            return preds.squeeze(0), A.squeeze(0)
        return preds, A

    def forward_test(self, args: TensorOrArgs):
        preds, _ = self.forward(args)
        return preds


class GrazianiRegression(CAMILBase):
    """
    Graziani et al. regression head:
        Flatten -> ReLU -> FC -> Dropout -> FC

    Notes
    -----
    - This follows the order shown in your figure.
    - If you want a more conventional order, you can change it to:
        Flatten -> FC -> ReLU -> Dropout -> FC
    - bounded=True is useful for PAM50 soft-label regression in [0,1]
    """

    def __init__(
        self,
        n_feats: int,
        n_out: int = 1,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        emb_dim: int = 256,
        attn_dim: Optional[int] = None,
        dropout: float = 0.25,
        head_hidden_dim: Optional[int] = None,
        head_dropout: float = 0.5,
        bounded: bool = False,
    ) -> None:
        super().__init__(
            n_feats=n_feats,
            encoder=encoder,
            attention=attention,
            emb_dim=emb_dim,
            attn_dim=attn_dim,
            dropout=dropout,
        )

        head_hidden_dim = head_hidden_dim or emb_dim

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, head_hidden_dim),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, n_out),
        )

        self.n_out = n_out
        self.bounded = bounded
        self.head_hidden_dim = head_hidden_dim
        self._init_linear_weights()

    def forward(self, args: TensorOrArgs):
        M, A, squeezed = self._encode_bag(args)
        preds = self.head(M)

        if self.bounded:
            preds = torch.sigmoid(preds)

        if squeezed:
            return preds.squeeze(0), A.squeeze(0)
        return preds, A

    def forward_test(self, args: TensorOrArgs):
        preds, _ = self.forward(args)
        return preds


# backward-compatible alias
# default alias keeps regression behavior
MILModel = CAMILRegression


if __name__ == "__main__":
    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Test input
    # ---------------------------------------------------------
    H = torch.rand(500, 2048)
    bags = torch.rand(2, 512, 2048)
    lens = torch.tensor([380, 512])

    # ---------------------------------------------------------
    # 1) CAMIL Classification
    # ---------------------------------------------------------
    cls_model = CAMILClassification(
        n_feats=2048,
        n_out=3,
        emb_dim=256,
        dropout=0.25,
        head_dropout=0.5,
        use_bn=True,
    )
    print("=== CAMILClassification ===")
    print(cls_model)

    logits, A = cls_model(H)
    assert logits.shape == (3,), logits.shape
    assert A.shape == (500,), A.shape
    assert abs(A.sum().item() - 1.0) < 1e-5

    probs, _ = cls_model.predict_proba(H)
    assert probs.shape == (3,), probs.shape
    assert abs(probs.sum().item() - 1.0) < 1e-5

    logits_b, A_b = cls_model((bags, lens))
    assert logits_b.shape == (2, 3), logits_b.shape
    assert A_b.shape == (2, 512), A_b.shape
    print("CAMILClassification tests passed.\n")

    # ---------------------------------------------------------
    # 2) CAMIL Regression
    # ---------------------------------------------------------
    reg_model = CAMILRegression(
        n_feats=2048,
        n_out=1,
        emb_dim=256,
        dropout=0.25,
        bounded=False,
    )
    print("=== CAMILRegression ===")
    print(reg_model)

    pred_r, A_r = reg_model(H)
    assert pred_r.shape == (1,), pred_r.shape
    assert A_r.shape == (500,), A_r.shape

    pred_rb, A_rb = reg_model((bags, lens))
    assert pred_rb.shape == (2, 1), pred_rb.shape
    assert A_rb.shape == (2, 512), A_rb.shape
    print("CAMILRegression tests passed.\n")

    # ---------------------------------------------------------
    # 3) Graziani Regression
    # ---------------------------------------------------------
    graz_model = GrazianiRegression(
        n_feats=2048,
        n_out=1,
        emb_dim=256,
        dropout=0.25,
        head_hidden_dim=256,
        head_dropout=0.5,
        bounded=False,
    )
    print("=== GrazianiRegression ===")
    print(graz_model)

    pred_g, A_g = graz_model(H)
    assert pred_g.shape == (1,), pred_g.shape
    assert A_g.shape == (500,), A_g.shape

    pred_gb, A_gb = graz_model((bags, lens))
    assert pred_gb.shape == (2, 1), pred_gb.shape
    assert A_gb.shape == (2, 512), A_gb.shape
    print("GrazianiRegression tests passed.\n")

    # ---------------------------------------------------------
    # 4) PAM50 soft-label regression examples
    # ---------------------------------------------------------
    pam50_camil = CAMILRegression(
        n_feats=2048,
        n_out=3,
        emb_dim=256,
        bounded=True,
    )
    pam50_graz = GrazianiRegression(
        n_feats=2048,
        n_out=3,
        emb_dim=256,
        head_hidden_dim=256,
        bounded=True,
    )

    preds_camil, _ = pam50_camil(H)
    preds_graz, _ = pam50_graz(H)

    assert preds_camil.shape == (3,), preds_camil.shape
    assert preds_graz.shape == (3,), preds_graz.shape
    assert torch.all(preds_camil >= 0) and torch.all(preds_camil <= 1)
    assert torch.all(preds_graz >= 0) and torch.all(preds_graz <= 1)

    print("PAM50 CAMIL soft-label regression test passed.")
    print("PAM50 Graziani soft-label regression test passed.\n")

    print("All tests passed.")
