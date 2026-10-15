# Chain source provenance

The implementation follows the paper's Chain decision rule and its stated precondition. Eq. (11) and the prose take precedence over the conflicting OR printed in the pseudocode's inner condition; the implementation makes one current-blocker relocation and does not add Chain F's fill correction.

- Primary paper: Raka Jovanovic and Stefan Voß, “A chain heuristic for the blocks relocation problem,” *Computers & Industrial Engineering* 75 (2014), pp. 79–86, DOI [10.1016/j.cie.2014.06.010](https://doi.org/10.1016/j.cie.2014.06.010).
- Local source: original local `参考论文/chain.pdf` (not redistributed); SHA256 `8a6b8624fd9f16e0440f65e99ceb74333364c7d830c1d309ace30047e568b655`. The decision definitions are in §3, especially Eq. (11) and its surrounding prose on PDF p. 4.
- Author-hosted paper: [BRP heuristic PDF](https://mail.ipb.ac.rs/~rakaj/home/brpheuristic.pdf); [Raka Jovanovic's home page](https://mail.ipb.ac.rs/~rakaj/home/).
- Additional institutional page checked: [University of Hamburg project listing](https://www.bwl.uni-hamburg.de/iwi/forschung/projekte.html). Its linked BRP code pointer is the Corridor Method, which is a different method and was not treated as Chain source code.
- Historical author-code lookup: the original bounded check covered the author-hosted paper and home page, the publisher DOI page, the Hamburg project page, and title/method-targeted GitHub searches. No author-maintained Chain implementation was located in these sources. This is a bounded negative result, not proof that no code exists elsewhere.

The offline implementation is [`chain_offline.py`](../_snapshot/Chain/chain_offline.py); the online one-step adapter is [`chain_public.py`](../_snapshot/Chain/chain_public.py). See [`README.md`](README.md) for frozen rule choices and [reproduction and verification instructions](../../docs/REPRODUCTION.md) for the shared correctness suites.
