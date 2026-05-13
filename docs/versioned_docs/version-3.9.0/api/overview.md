---
slug: /apioverview
displayed_sidebar: apisidebar
title: 'API Overview'
---

# ICICLE APIs

This section provides detailed documentation for ICICLE’s core APIs in this **open-icicle** distribution — MSM, NTT, ECNTT, and supporting vector operations. Each API includes dedicated examples for C++, Go, and Rust—choose your preferred language to get started.

:::important Scope
Only the **BN254** and **BLS12-381** curves are supported. Primitives outside MSM / NTT / ECNTT / vector ops (e.g. Poseidon, Merkle trees, polynomial APIs, sumcheck, FRI, hash, pairings) are not part of this distribution.
:::

import Link from '@docusaurus/Link';

<div className="card-row-3">

  <Link to="/cppstart" className="card-link">
    <div className="card-box icon-only">
      <img alt="C++" className="card-icon-top cpp-icon" />
    </div>
  </Link>

  <Link to="/gooverview" className="card-link">
    <div className="card-box icon-only">
      <img alt="Golang" className="card-icon-top go-icon" />
    </div>
  </Link>

  <Link to="/rustoverview" className="card-link">
    <div className="card-box icon-only">
      <img alt="Rust" className="card-icon-top rust-icon" />
    </div>
  </Link>
</div>