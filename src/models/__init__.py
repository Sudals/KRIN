"""Reconstruction models — all share the signature

    forward(hist, x_obs, obs_mask) -> pred        # all in scaled space

where ``hist`` is (B, K, N), ``x_obs`` is the target-day observed values with
hidden positions zeroed (B, N), and ``obs_mask`` is (B, N) with 1 = observed.
"""
from .networks import (STGNN, IGNNK, RNNKriging, STGNNRobust, STGNNRobustTV,
                       STGNNRobustMR, STGNNRobustIA, GATGRU, STGNNRobustGP,
                       STGNNRobustGP2, STGNNRobustW, KRIN, STGNNRevIN,
                       LinearKriging, KRINNoCenter, KRINStd, Centered, KRINv2, KRINAnchored, KRINSupport, KRINAttn, KRINTemporalProbe, KRINHom,
                       CenteredLast, CenteredStd)
from .sota import GRINBi, SATCN, GRIN, SPIN, KITS


def build_model(name: str, n_nodes: int, K: int, cfg: dict):
    name = name.lower()
    h = cfg["train"]["hidden_dim"]
    L = cfg["train"]["num_layers"]
    p = cfg["train"]["dropout"]
    if name == "stgnn":
        return STGNN(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name in ("stgnnr", "stgnn_robust"):
        return STGNNRobust(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name in ("stgnnrtv", "stgnnr_tv"):
        return STGNNRobustTV(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name in ("stgnnrmr", "stgnnr_mr"):
        return STGNNRobustMR(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name in ("stgnnria", "stgnnr_ia"):
        return STGNNRobustIA(n_nodes, K, hidden=h, layers=L, dropout=p)
    if name == "gatgru":
        return GATGRU(n_nodes, K, hidden=h, layers=max(L, 2), dropout=p)
    if name in ("stgnnrgp", "stgnnr_gp"):
        return STGNNRobustGP(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name in ("stgnnrw", "stgnnr_w"):
        return STGNNRobustW(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name == "krin":
        return KRIN(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name in ("stgnnrgp2", "stgnnr_gp2"):
        return STGNNRobustGP2(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name == "ignnk":
        return IGNNK(n_nodes, K, hidden=h, gconv_layers=max(L, 3), dropout=p)
    if name in ("linear", "mlp"):
        # Graph-free controls: the simplest learned component KRIN can wrap.
        # `linear` is a single affine map and `mlp` a two-hidden-layer readout,
        # both applied per node with no exchange of information between nodes.
        return LinearKriging(n_nodes, K, hidden=h, dropout=p, deep=(name == "mlp"))
    if name in ("gru", "lstm"):
        return RNNKriging(n_nodes, K, hidden=h, cell=name, dropout=p)
    if name == "satcn":
        return SATCN(n_nodes, K, hidden=h, gconv_layers=max(L, 3), dropout=p)
    if name.endswith("_off"):
        # Faithful ports of the published implementations, at the authors'
        # own defaults. Capacity is deliberately NOT matched here; see the
        # appendix on the fidelity arm.
        from .official_wrap import (IGNNKOfficial, SATCNOfficial, GRINOfficial,
                                    SPINOfficial, KITSOfficial)
        OFF = {"ignnk_off": IGNNKOfficial, "satcn_off": SATCNOfficial,
               "grin_off": GRINOfficial, "spin_off": SPINOfficial,
               "kits_off": KITSOfficial}
        return OFF[name](n_nodes, K)
    if name == "grin_wide":
        # Capacity control for grin_bi: unidirectional GRIN widened to the same
        # parameter budget (45,841 vs 46,273 on the subway), so the bidirectional
        # arm is not just "more parameters".
        return GRIN(n_nodes, K, hidden=80, gconv_layers=L, dropout=p)
    if name in ("grin_bi", "grinbi"):
        return GRINBi(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name == "grin":
        return GRIN(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name == "spin":
        return SPIN(n_nodes, K, hidden=h, dropout=p)
    if name == "kits":
        return KITS(n_nodes, K, hidden=h, gconv_layers=max(L, 3), dropout=p)
    # --- ablation variants: centring on/off, and full mean/variance ----------
    if name in ("krin_noc", "krinnoc"):
        return KRINNoCenter(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name == "krinat":          # centring + attention aggregator + rho-weighted anchor
        return KRINAttn(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    # capacity sweep on our own model. Parameter parity with the baselines is not
    # required for the main table -- and in this setting capacity does not predict
    # accuracy anyway (the largest baseline, SPIN at 54,913, is among the worst on
    # airports; the smallest, IGNNK at 17,793, is the best). These variants tell us
    # whether OUR model has headroom left.
    if name == "krin128":
        return KRIN(n_nodes, K, hidden=128, gconv_layers=L, dropout=p)
    if name == "krinat128":
        return KRINAttn(n_nodes, K, hidden=128, gconv_layers=L, dropout=p)
    if name == "krinh":           # location by subtraction, scale by homogeneity
        return KRINHom(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name in ("krin_lin", "krin_not"):
        return KRINTemporalProbe(n_nodes, K, hidden=h, gconv_layers=L, dropout=p,
                                 mode={"krin_lin": "lin", "krin_not": "none"}[name])
    if name == "krinat_noa":      # ablation: no spatial anchor (attention + centring)
        return KRINAttn(n_nodes, K, hidden=h, gconv_layers=L, dropout=p, use_anchor=False)
    if name == "krinat_noc":      # ablation: no centring (attention + anchor)
        return KRINAttn(n_nodes, K, hidden=h, gconv_layers=L, dropout=p, use_centring=False)
    if name == "krinat_none":     # ablation: neither (plain attention model)
        return KRINAttn(n_nodes, K, hidden=h, gconv_layers=L, dropout=p,
                        use_anchor=False, use_centring=False)
    if name == "krinat_d4":       # deeper attention stack
        return KRINAttn(n_nodes, K, hidden=h, gconv_layers=L, dropout=p, blocks=4)
    if name == "krins":           # support-aware: rho-weighted anchor + support gate
        return KRINSupport(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name == "krina":           # spatial anchor as a parameter-free additive path
        return KRINAnchored(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name == "krinv2":          # both changes
        return KRINv2(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name == "krinv2c":         # periodicity-matched centring only (keeps GRU)
        return KRINv2(n_nodes, K, hidden=h, gconv_layers=L, dropout=p, recurrent=True)
    if name == "krinv2g":         # recurrence removed only (flat centring)
        return KRINv2(n_nodes, K, hidden=h, gconv_layers=L, dropout=p, use_dow=False)
    if name in ("krin_std", "krinstd"):
        return KRINStd(n_nodes, K, hidden=h, gconv_layers=L, dropout=p)
    if name.endswith("_cd"):         # baseline + the SAME periodicity-matched centring
        return Centered(build_model(name[:-3], n_nodes, K, cfg), K, use_dow=True)
    if name.endswith("_cl"):         # last-value centring (NLinear-style)
        return CenteredLast(build_model(name[:-3], n_nodes, K, cfg), K, use_dow=False)
    if name.endswith("_cs"):         # mean + std (full RevIN)
        return CenteredStd(build_model(name[:-3], n_nodes, K, cfg), K, use_dow=False)
    if name.endswith("_c"):          # e.g. "ignnk_c" = IGNNK + plain window-mean centring
        return Centered(build_model(name[:-2], n_nodes, K, cfg), K, use_dow=False)
    raise ValueError(f"unknown model: {name}")
