import inspect

import gradio as gr

# ── Palette ────────────────────────────────────────────────────────────────
BG = "#0a0a0b"
SURFACE = "#111113"
SURFACE_2 = "#18181b"
BORDER = "#222226"
BORDER_STRONG = "#2e2e33"
TEXT = "#ededef"
MUTED = "#86868f"
ACCENT = "#c8f031"
ACCENT_HOVER = "#d6f75e"


def _both(**values: str) -> dict[str, str]:
    """Duplicate every theme variable into its `_dark` variant (forced single look)."""
    valid = inspect.signature(gr.themes.Base.set).parameters
    out: dict[str, str] = {}
    for key, val in values.items():
        out[key] = val
        if f"{key}_dark" in valid:
            out[f"{key}_dark"] = val
    return out


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue="lime",
        neutral_hue="zinc",
        radius_size="sm",
        font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"),
        font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace"),
    )
    return theme.set(
        **_both(
            body_background_fill=BG,
            body_text_color=TEXT,
            body_text_color_subdued=MUTED,
            background_fill_primary=SURFACE,
            background_fill_secondary=BG,
            border_color_primary=BORDER,
            border_color_accent=ACCENT,
            color_accent=ACCENT,
            color_accent_soft="rgba(200, 240, 49, 0.12)",
            shadow_drop="none",
            shadow_drop_lg="none",
            shadow_spread="0px",
            block_background_fill="transparent",
            block_border_width="0px",
            block_shadow="none",
            block_padding="0px",
            block_label_background_fill="transparent",
            block_title_background_fill="transparent",
            block_label_text_color=MUTED,
            block_title_text_color=MUTED,
            block_info_text_color=MUTED,
            panel_background_fill=SURFACE,
            panel_border_color=BORDER,
            input_background_fill=SURFACE,
            input_background_fill_hover=SURFACE,
            input_background_fill_focus=SURFACE,
            input_border_color=BORDER_STRONG,
            input_border_color_hover="#3a3a40",
            input_border_color_focus=ACCENT,
            input_shadow="none",
            input_shadow_focus="none",
            checkbox_label_background_fill=BG,
            checkbox_label_background_fill_hover=SURFACE_2,
            checkbox_label_background_fill_selected=TEXT,
            checkbox_label_border_color=BORDER_STRONG,
            checkbox_label_border_color_selected=TEXT,
            checkbox_label_text_color=MUTED,
            checkbox_label_text_color_selected=BG,
            checkbox_label_shadow="none",
            checkbox_background_color_selected=BG,
            checkbox_border_color_selected=BG,
            slider_color=ACCENT,
            loader_color=ACCENT,
            button_primary_background_fill=ACCENT,
            button_primary_background_fill_hover=ACCENT_HOVER,
            button_primary_border_color=ACCENT,
            button_primary_border_color_hover=ACCENT_HOVER,
            button_primary_text_color=BG,
            button_primary_text_color_hover=BG,
            button_primary_shadow="none",
            button_secondary_background_fill=SURFACE_2,
            button_secondary_background_fill_hover="#202024",
            button_secondary_border_color=BORDER_STRONG,
            button_secondary_border_color_hover="#3a3a40",
            button_secondary_text_color=TEXT,
            button_secondary_text_color_hover=TEXT,
            button_secondary_shadow="none",
            button_transform_hover="none",
            button_transform_active="scale(0.98)",
        ),
        block_label_text_size="11px",
        block_title_text_size="11px",
        block_title_text_weight="500",
        block_label_text_weight="500",
        button_large_text_weight="600",
        button_medium_text_weight="500",
        input_border_width="1px",
        layout_gap="16px",
        form_gap_width="0px",
    )


CUSTOM_CSS = f"""
/* ── Layout ─────────────────────────────────────────────────────────── */
.gradio-container {{
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 28px 32px 48px !important;
}}
footer {{ display: none !important; }}

/* Gradio draws separators by painting .form with the border colour — drop it */
.gradio-container .form {{
    background: transparent !important;
    border: none !important;
    gap: 14px !important;
}}
.gradio-container .block {{ background: transparent !important; }}

/* Labels: small, uppercase, tracked — Swiss-style hierarchy */
.gradio-container label > span,
.gradio-container .block-title,
.gradio-container [data-testid="block-info"] {{
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 11px !important;
    color: {MUTED} !important;
}}

/* ── Header ─────────────────────────────────────────────────────────── */
.app-header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid {BORDER};
}}
.app-header .brand {{ display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }}
.app-header h1 {{
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin: 0;
    color: {TEXT};
}}
.app-header h1 span {{ color: {MUTED}; font-weight: 400; }}
.app-header p {{ margin: 0; color: {MUTED}; font-size: 13px; }}
.kbd-hints {{ display: flex; gap: 16px; color: {MUTED}; font-size: 12px; }}
kbd {{
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 11px;
    padding: 2px 6px;
    border: 1px solid {BORDER_STRONG};
    border-bottom-width: 2px;
    border-radius: 4px;
    color: {TEXT};
    margin-right: 6px;
}}

/* ── Panels ─────────────────────────────────────────────────────────── */
.panel {{ gap: 14px !important; }}
.panel + .panel {{ margin-top: 20px; }}
.section-title {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {MUTED};
    display: flex;
    align-items: center;
    gap: 10px;
}}
.section-title::after {{
    content: "";
    flex: 1;
    height: 1px;
    background: {BORDER};
}}

/* Segmented control for algorithm radio */
.segmented .wrap {{ gap: 0 !important; display: grid !important; grid-template-columns: repeat(4, 1fr); }}
.segmented label {{
    justify-content: center !important;
    border-radius: 0 !important;
    margin: 0 !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    letter-spacing: 0.04em;
    padding: 8px 0 !important;
    border: 1px solid {BORDER_STRONG} !important;
    background: {SURFACE} !important;
    color: {MUTED} !important;
    cursor: pointer;
    transition: background 120ms ease, color 120ms ease;
}}
.segmented label:hover {{ color: {TEXT} !important; }}
.segmented label.selected,
.segmented label:has(input:checked) {{
    background: {TEXT} !important;
    border-color: {TEXT} !important;
    color: {BG} !important;
}}
.segmented label span {{ color: inherit !important; }}
.segmented label:first-child {{ border-radius: 6px 0 0 6px !important; }}
.segmented label:last-child {{ border-radius: 0 6px 6px 0 !important; }}
.segmented label:not(:first-child) {{ border-left-width: 0 !important; }}
.segmented input[type="radio"] {{ display: none !important; }}

/* Buttons */
.gradio-container button {{ transition: background 120ms ease, border-color 120ms ease, transform 80ms ease; }}
#step-btn {{ height: 48px; font-size: 15px !important; letter-spacing: 0.01em; }}

/* Model status line */
.model-status {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: {MUTED};
    font-family: "JetBrains Mono", ui-monospace, monospace;
    word-break: break-all;
}}
.model-status b {{ color: {TEXT}; font-weight: 500; }}

/* ── Dots ───────────────────────────────────────────────────────────── */
.dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
.dot.ok {{ background: #4ade80; box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.15); }}
.dot.warn {{ background: #f5a524; box-shadow: 0 0 0 3px rgba(245, 165, 36, 0.15); }}
.dot.bad {{ background: #f26d6d; box-shadow: 0 0 0 3px rgba(242, 109, 109, 0.15); }}

/* ── HUD ────────────────────────────────────────────────────────────── */
.hud {{
    display: flex;
    flex-wrap: wrap;
    gap: 0;
    border: 1px solid {BORDER};
    border-radius: 12px;
    background: {SURFACE};
    overflow: hidden;
}}
.hud-cell {{
    padding: 14px 20px;
    border-right: 1px solid {BORDER};
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 110px;
}}
.hud-cell:last-child {{ border-right: none; }}
.hud-cell.grow {{ flex: 1; }}
.hud-label {{
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {MUTED};
}}
.hud-label sub {{ text-transform: none; }}
.hud-value {{
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 20px;
    font-weight: 500;
    color: {TEXT};
    line-height: 1.1;
    display: flex;
    align-items: center;
    gap: 8px;
    font-variant-numeric: tabular-nums;
}}
.hud-value.small {{ font-size: 13px; font-family: Inter, ui-sans-serif, sans-serif; }}
.pos {{ color: {ACCENT} !important; }}
.neg {{ color: #f26d6d !important; }}

.vitals {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 6px 24px; }}
.vital {{ display: flex; align-items: center; gap: 10px; }}
.vital-name {{ font-size: 12px; color: {MUTED}; width: 58px; }}
.pips {{ display: flex; gap: 3px; }}
.pip {{ width: 8px; height: 14px; border-radius: 2px; background: {BORDER_STRONG}; }}
.pip.on {{ background: {TEXT}; }}
.vital.low .pip.on {{ background: #f26d6d; }}
.vital-num {{
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 12px;
    color: {MUTED};
    font-variant-numeric: tabular-nums;
}}

/* ── Stage (image matrix) ───────────────────────────────────────────── */
.caption {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 0 2px 8px;
}}
.caption .sym {{
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 14px;
    color: {TEXT};
}}
.caption .desc {{ font-size: 12px; color: {MUTED}; }}
.gradio-container .frame {{
    background: {BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    aspect-ratio: 1 / 1;
}}
.frame img {{
    image-rendering: pixelated !important;
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
}}
.frame.model {{ border-color: rgba(200, 240, 49, 0.22) !important; }}

/* ── Metric cards ───────────────────────────────────────────────────── */
.metric-card {{ display: flex; flex-direction: column; gap: 16px; }}
.action-name {{
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 22px;
    font-weight: 500;
    color: {TEXT};
    letter-spacing: -0.01em;
    display: flex;
    align-items: baseline;
    gap: 10px;
}}
.action-name small {{ font-size: 12px; color: {MUTED}; font-weight: 400; }}
.kv {{ display: flex; flex-direction: column; gap: 12px; }}
.kv-row {{ display: flex; justify-content: space-between; align-items: baseline; }}
.kv-key {{ font-size: 12px; color: {MUTED}; }}
.kv-key i {{ font-family: "JetBrains Mono", ui-monospace, monospace; font-style: normal; color: #b4b4bb; }}
.kv-val {{
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 15px;
    color: {TEXT};
    font-variant-numeric: tabular-nums;
}}
.bar {{ height: 4px; background: {BORDER_STRONG}; border-radius: 2px; overflow: hidden; }}
.bar > div {{ height: 100%; background: {TEXT}; border-radius: 2px; transition: width 200ms ease; }}
.bar.ok > div {{ background: {ACCENT}; }}
.bar.warn > div {{ background: #f5a524; }}
.bar.bad > div {{ background: #f26d6d; }}
.big-num {{
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 30px;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: {TEXT};
    font-variant-numeric: tabular-nums;
    line-height: 1;
}}
.chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 8px;
    border: 1px solid {BORDER_STRONG};
    border-radius: 999px;
    color: {TEXT};
}}
.note {{ font-size: 13px; line-height: 1.55; color: {MUTED}; }}
.note code {{
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 12px;
    color: {TEXT};
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    padding: 1px 6px;
    border-radius: 4px;
}}
.empty {{ font-size: 13px; color: {MUTED}; padding: 4px 0; }}
"""

KEYBOARD_JS = """
function() {
    document.body.classList.add('dark');
    document.addEventListener('keydown', function(e) {
        const tag = document.activeElement ? document.activeElement.tagName : '';
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;
        let id = null;
        if (e.code === 'Space' || e.code === 'ArrowRight') id = 'step-btn';
        else if (e.code === 'KeyP') id = 'play-btn';
        else if (e.code === 'KeyR') id = 'reset-btn';
        if (!id) return;
        const btn = document.getElementById(id);
        if (btn) {
            e.preventDefault();
            btn.click();
        }
    });
}
"""
