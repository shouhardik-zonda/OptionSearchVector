# option_search_app.py
"""
Streamlit UI for local vector search using ONLY:
  - getEmbeddings(text)
  - optimizedLocalVectorHeapSearch(query_embedding, top_k=..., batch_size=...)

Fixes:
- Cast ObjectId and other non-primitive values to strings before DataFrame display.
- Use json.dumps(..., default=str) for download + detail views to avoid ujson/Arrow issues.
"""

import os
import sys
import time
import json
from typing import Any, Dict, List

import streamlit as st
import pandas as pd

# --- Import your functions from vector.py ---
try:
    from vector import getEmbeddings, optimizedLocalVectorHeapSearch
except Exception:
    sys.path.append(os.getcwd())
    from vector import getEmbeddings, optimizedLocalVectorHeapSearch

st.set_page_config(page_title="Option Search — Vector Similarity", layout="wide")
st.title("🔎 Option Search — Vector Similarity (Heap, Unique by Product)")

with st.sidebar:
    st.header("Settings")
    default_query = st.text_input("Default query", value="Echo Spot")
    top_k = st.slider("Top K results", 1, 100, 10, 1)
    batch_size = st.slider("Batch size (DB fetch / compute)", 128, 8192, 1000, 128)
    st.caption("This UI calls your `optimizedLocalVectorHeapSearch` directly.")

query = st.text_input(
    "Enter your query",
    value=default_query,
    help="Type a natural language query (e.g., product or feature description).",
)

def run_search(q: str):
    q = (q or "").strip()
    if not q:
        return [], 0.0
    q_emb = getEmbeddings(q)
    t0 = time.time()
    results = optimizedLocalVectorHeapSearch(q_emb, top_k=top_k, batch_size=batch_size)
    latency = time.time() - t0
    return results, latency

def safe_str(x: Any) -> Any:
    """Return primitives as-is; stringify anything non-JSON-native (e.g., ObjectId)."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    try:
        return str(x)
    except Exception:
        return repr(x)

def flatten_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one result for table display (only primitive columns)."""
    product = r.get("product") or {}
    option = r.get("option") or {}
    brand = product.get("brand") or {}
    category = product.get("category") or {}

    brand_name = brand.get("name") if isinstance(brand, dict) else brand
    category_name = category.get("name") if isinstance(category, dict) else category

    pid = (
        r.get("product_id")
        or product.get("id")
        or product.get("_id")
        or product.get("productId")
    )

    row = {
        "product_id": safe_str(pid),
        "product_name": safe_str(product.get("name")),
        "product_number": safe_str(product.get("number")),
        "brand": safe_str(brand_name),
        "category": safe_str(category_name),
        "option_id": safe_str(option.get("id") if isinstance(option, dict) else None),
        "option_name": safe_str(option.get("name") if isinstance(option, dict) else None),
        "score": float(r.get("score")) if r.get("score") is not None else None,
    }
    return row

if st.button("Search", type="primary"):
    with st.spinner("Computing similarities..."):
        results, latency = run_search(query)

    st.subheader(f"Top {len(results)} results for: “{query}” • {latency:.2f}s")

    if not results:
        st.info("No results. Try a different query.")
    else:
        # Table view (Arrow-safe: all columns are primitives / strings)
        flat = [flatten_row(r) for r in results]
        df = pd.DataFrame(flat)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Detail expanders (render JSON with default=str to handle ObjectId)
        for i, r in enumerate(results, 1):
            product = r.get("product") or {}
            option = r.get("option") or {}
            prod_name = product.get("name") or "Product"
            with st.expander(f"{i}. {prod_name} — score: {r.get('score')}"):
                st.markdown("**Product details**")
                st.code(json.dumps(product, indent=2, ensure_ascii=False, default=str), language="json")
                st.markdown("**Option details (best-matching option)**")
                st.code(json.dumps(option, indent=2, ensure_ascii=False, default=str), language="json")

        # Download JSON (avoid pandas/ujson; use stdlib json)
        payload = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "⬇️ Download results (JSON)",
            data=payload,
            file_name=f"option_search_{int(time.time())}.json",
            mime="application/json",
        )

st.markdown(
    """
---
_If imports fail, make sure `option_search_app.py` sits next to `vector.py` and that
`vector.py` defines `getEmbeddings` and `optimizedLocalVectorHeapSearch`._
"""
)
