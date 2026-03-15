"""
Admin page — review, approve, edit, and manage business-context suggestions.

This page is accessible from the Streamlit navigation sidebar as a separate tab.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.business_context import (
    get_suggestions,
    get_approved_context,
    approve_suggestion,
    reject_suggestion,
    remove_approved,
    update_approved_text,
    add_suggestion,
)

st.set_page_config(
    page_title="Context Manager — Admin",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Business Context Manager")
st.caption(
    "Review user-submitted business context suggestions and manage the "
    "approved context that gets injected into the analytics assistant's prompts."
)

# ─────────────────────────────────────────────────────────────────────────────
# Pending Suggestions
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("Pending Suggestions")

pending = get_suggestions(status="pending")

if not pending:
    st.info("No pending suggestions to review. All caught up!")
else:
    st.markdown(f"**{len(pending)}** suggestion(s) awaiting review.")
    for suggestion in pending:
        with st.container(border=True):
            col_text, col_actions = st.columns([3, 1])

            with col_text:
                st.markdown(f"**Suggestion** `{suggestion['id']}`")
                st.markdown(f"> {suggestion['text']}")
                st.caption(
                    f"By **{suggestion.get('submitted_by', 'user')}** "
                    f"on {suggestion['submitted_at'][:10]}"
                )

            with col_actions:
                edited = st.text_area(
                    "Edit before approving (optional)",
                    value=suggestion["text"],
                    key=f"edit_{suggestion['id']}",
                    height=80,
                )

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button(
                        "✅ Approve",
                        key=f"approve_{suggestion['id']}",
                        use_container_width=True,
                    ):
                        final_text = edited if edited.strip() != suggestion["text"] else None
                        approve_suggestion(suggestion["id"], admin_text=final_text)
                        st.toast(f"Approved suggestion {suggestion['id']}", icon="✅")
                        st.rerun()

                with btn_col2:
                    if st.button(
                        "❌ Reject",
                        key=f"reject_{suggestion['id']}",
                        use_container_width=True,
                        type="secondary",
                    ):
                        reject_suggestion(suggestion["id"])
                        st.toast(f"Rejected suggestion {suggestion['id']}", icon="❌")
                        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Approved Context (active)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("Approved Business Context")
st.caption(
    "These entries are actively injected into every LLM prompt. "
    "You can edit or remove them."
)

approved = get_approved_context()

if not approved:
    st.info(
        "No approved context yet. Approve suggestions above, or add your own below."
    )
else:
    for entry in approved:
        with st.container(border=True):
            col_text, col_actions = st.columns([3, 1])

            with col_text:
                st.markdown(f"**Context** `{entry['id']}`")
                st.markdown(f"{entry['text']}")
                meta_parts = []
                if entry.get("submitted_by"):
                    meta_parts.append(f"Submitted by **{entry['submitted_by']}**")
                if entry.get("approved_at"):
                    meta_parts.append(f"Approved {entry['approved_at'][:10]}")
                if entry.get("original_text") and entry["original_text"] != entry["text"]:
                    meta_parts.append("_(edited by admin)_")
                if meta_parts:
                    st.caption(" · ".join(meta_parts))

            with col_actions:
                new_text = st.text_area(
                    "Edit text",
                    value=entry["text"],
                    key=f"edit_approved_{entry['id']}",
                    height=80,
                )
                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    if st.button(
                        "💾 Save",
                        key=f"save_{entry['id']}",
                        use_container_width=True,
                    ):
                        if new_text.strip() and new_text.strip() != entry["text"]:
                            update_approved_text(entry["id"], new_text.strip())
                            st.toast("Context updated", icon="💾")
                            st.rerun()
                        else:
                            st.toast("No changes to save", icon="ℹ️")
                with btn_c2:
                    if st.button(
                        "🗑️ Remove",
                        key=f"remove_{entry['id']}",
                        use_container_width=True,
                        type="secondary",
                    ):
                        remove_approved(entry["id"])
                        st.toast(f"Removed context {entry['id']}", icon="🗑️")
                        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Add context directly (admin shortcut)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("Add Context Directly")
st.caption(
    "As an admin, you can add business context directly without going "
    "through the suggestion/approval flow."
)

direct_text = st.text_area(
    "Business context to add",
    key="admin_direct_ctx",
    height=100,
    placeholder=(
        'e.g. "Cancellation rate" = cancelled_consult_orders / total_consult_orders. '
        "Free follow-ups should be excluded from revenue calculations."
    ),
)
if st.button("Add to approved context", use_container_width=True, type="primary"):
    if direct_text and direct_text.strip():
        entry = add_suggestion(direct_text.strip(), submitted_by="admin")
        approve_suggestion(entry["id"])
        st.toast("Added to approved context", icon="✅")
        st.rerun()
    else:
        st.warning("Please enter context text.")

# ─────────────────────────────────────────────────────────────────────────────
# Review history
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("Review history (all suggestions)", expanded=False):
    all_suggestions = get_suggestions()
    if all_suggestions:
        for s in reversed(all_suggestions):
            status_icon = {"pending": "🟡", "approved": "🟢", "rejected": "🔴"}.get(
                s.get("status", ""), "⚪"
            )
            st.markdown(
                f"{status_icon} **{s['status'].upper()}** `{s['id']}` — "
                f"{s['text'][:80]}{'…' if len(s['text']) > 80 else ''}"
            )
            st.caption(
                f"By {s.get('submitted_by', 'user')} · "
                f"{s['submitted_at'][:10]}"
                + (f" · Reviewed {s['reviewed_at'][:10]}" if s.get("reviewed_at") else "")
            )
    else:
        st.caption("No suggestions have been submitted yet.")
