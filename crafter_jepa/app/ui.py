import gradio as gr
from PIL import Image

from crafter_jepa.app.constants import (
    BLANK_IMAGE,
    CRAFTER_ACTIONS,
    find_checkpoints,
    obs_to_img,
    tensor_to_img,
)
from crafter_jepa.app.env_session import CrafterSession, EnvSnapshot
from crafter_jepa.app.formatters import (
    format_decision,
    format_decision_empty,
    format_model_status,
    format_prediction_info,
    format_vitals,
)
from crafter_jepa.app.model_session import ModelSession, StepPrediction
from crafter_jepa.app.styles import CUSTOM_CSS, KEYBOARD_JS, build_theme

AGENT_CHOICE = "Agent (polityka)"
PLAY_LABEL = "Odtwarzaj"
PAUSE_LABEL = "Pauza"


class AppState:
    def __init__(self) -> None:
        self.env_session = CrafterSession(seed=42)
        self.model_session = ModelSession(algo="jepa")
        self.last_snapshot = EnvSnapshot(
            prev_obs=self.env_session.obs,
            next_obs=self.env_session.obs,
            reward=0.0,
            done=False,
            step_num=0,
            total_reward=0.0,
        )
        self.last_prediction = StepPrediction(
            recon_curr=None,
            pred_next=None,
            mse_val=None,
            v_est=0.0,
            act_prob=1.0,
            act_name="noop",
            act_idx=0,
            algo="jepa",
        )


state = AppState()

StepOutputs = tuple[
    str,
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    str,
    str,
]


def on_change_model(algo: str, checkpoint_path: str) -> tuple[str, str]:
    state.model_session = ModelSession(algo=algo, checkpoint_path=checkpoint_path)
    state.last_prediction.algo = state.model_session.algo
    status_html = format_model_status(
        state.model_session.algo, state.model_session.loaded_ckpt
    )
    return status_html, format_prediction_info(state.last_prediction)


def on_step(action_choice: str) -> StepOutputs:
    if state.env_session.done:
        state.env_session.reset()

    curr_obs = state.env_session.obs

    if action_choice == AGENT_CHOICE:
        act_idx, _, _ = state.model_session.select_agent_action(curr_obs)
    else:
        act_idx = CRAFTER_ACTIONS.index(action_choice)

    snapshot = state.env_session.step(act_idx)
    state.last_snapshot = snapshot

    prediction = state.model_session.process_step(
        obs_t_np=snapshot.prev_obs,
        action=act_idx,
        next_obs_np=snapshot.next_obs,
    )
    state.last_prediction = prediction

    return (
        format_vitals(snapshot),
        obs_to_img(snapshot.prev_obs),
        tensor_to_img(prediction.recon_curr),
        obs_to_img(snapshot.next_obs),
        tensor_to_img(prediction.pred_next),
        format_decision(prediction, snapshot.reward),
        format_prediction_info(prediction),
    )


def on_reset() -> StepOutputs:
    obs = state.env_session.reset()
    state.last_snapshot = EnvSnapshot(
        prev_obs=obs,
        next_obs=obs,
        reward=0.0,
        done=False,
        step_num=0,
        total_reward=0.0,
    )
    blank = BLANK_IMAGE.resize((256, 256), Image.Resampling.NEAREST)

    return (
        format_vitals(state.last_snapshot),
        obs_to_img(obs),
        blank,
        blank,
        blank,
        format_decision_empty(),
        format_prediction_info(state.last_prediction),
    )


HEADER_HTML = """
<header class="app-header">
  <div class="brand">
    <h1>Crafter <span>/ Latent Explorer</span></h1>
    <p>JEPA · VAE · SAE · PPO</p>
  </div>
  <div class="kbd-hints">
    <span><kbd>Spacja</kbd>krok</span>
    <span><kbd>P</kbd>odtwarzaj</span>
    <span><kbd>R</kbd>reset</span>
  </div>
</header>
"""


def _section(title: str) -> None:
    gr.HTML(f'<div class="section-title">{title}</div>')


def _caption(symbol: str, desc: str) -> None:
    gr.HTML(
        f'<div class="caption"><span class="sym">{symbol}</span>'
        f'<span class="desc">{desc}</span></div>'
    )


def _frame(value: Image.Image, model: bool = False) -> gr.Image:
    return gr.Image(
        value=value,
        show_label=False,
        container=False,
        interactive=False,
        buttons=[],
        format="png",
        elem_classes=["frame", "model"] if model else ["frame"],
    )


def build_app() -> gr.Blocks:
    ckpts = find_checkpoints()
    default_ckpt = ckpts[0] if ckpts else ""

    with gr.Blocks(title="Crafter · Latent Explorer") as app:
        gr.HTML(HEADER_HTML)

        vitals_display = gr.HTML(value=format_vitals(state.last_snapshot))

        with gr.Row(equal_height=False):
            # ── Sidebar: inputs ───────────────────────────────────────────
            with gr.Column(scale=0, min_width=300):
                with gr.Column(elem_classes=["panel"]):
                    _section("Model")
                    algo_radio = gr.Radio(
                        choices=["JEPA", "VAE", "SAE", "PPO"],
                        value="JEPA",
                        show_label=False,
                        container=False,
                        elem_classes=["segmented"],
                    )
                    ckpt_input = gr.Dropdown(
                        choices=ckpts,
                        value=default_ckpt,
                        label="Checkpoint",
                        allow_custom_value=True,
                    )
                    load_btn = gr.Button("Załaduj", variant="secondary", size="md")
                    load_status = gr.HTML(
                        value=format_model_status(
                            state.model_session.algo, state.model_session.loaded_ckpt
                        )
                    )

                with gr.Column(elem_classes=["panel"]):
                    _section("Sterowanie")
                    action_dropdown = gr.Dropdown(
                        choices=[AGENT_CHOICE] + CRAFTER_ACTIONS,
                        value=AGENT_CHOICE,
                        label="Źródło akcji",
                    )
                    step_btn = gr.Button("Krok", variant="primary", elem_id="step-btn")
                    with gr.Row():
                        play_btn = gr.Button(
                            PLAY_LABEL,
                            variant="secondary",
                            size="md",
                            min_width=100,
                            elem_id="play-btn",
                        )
                        reset_btn = gr.Button(
                            "↺  Reset",
                            variant="secondary",
                            size="md",
                            min_width=100,
                            elem_id="reset-btn",
                        )
                    fps_slider = gr.Slider(
                        minimum=0.5,
                        maximum=8.0,
                        value=2.0,
                        step=0.5,
                        label="Kroki / s",
                    )

            # ── Stage: observation vs model ──────────────────────────────
            with gr.Column(scale=3, min_width=520):
                with gr.Row():
                    with gr.Column(min_width=220):
                        _caption("o(t)", "środowisko")
                        img_real_t = _frame(obs_to_img(state.env_session.obs))
                    with gr.Column(min_width=220):
                        _caption("ô(t)", "model · rekonstrukcja")
                        img_recon_t = _frame(BLANK_IMAGE, model=True)
                with gr.Row():
                    with gr.Column(min_width=220):
                        _caption("o(t+1)", "środowisko · po akcji")
                        img_real_next = _frame(BLANK_IMAGE)
                    with gr.Column(min_width=220):
                        _caption("ô(t+1)", "model · predykcja")
                        img_pred_next = _frame(BLANK_IMAGE, model=True)

            # ── Metrics ──────────────────────────────────────────────────
            with gr.Column(scale=0, min_width=300):
                with gr.Column(elem_classes=["panel"]):
                    decision_display = gr.HTML(value=format_decision_empty())
                with gr.Column(elem_classes=["panel"]):
                    prediction_display = gr.HTML(
                        value=format_prediction_info(state.last_prediction)
                    )

        timer = gr.Timer(value=0.5, active=False)
        playing = gr.State(False)

        # ── Wiring ───────────────────────────────────────────────────────
        outputs_step = [
            vitals_display,
            img_real_t,
            img_recon_t,
            img_real_next,
            img_pred_next,
            decision_display,
            prediction_display,
        ]

        load_btn.click(
            fn=on_change_model,
            inputs=[algo_radio, ckpt_input],
            outputs=[load_status, prediction_display],
        )
        algo_radio.change(
            fn=on_change_model,
            inputs=[algo_radio, ckpt_input],
            outputs=[load_status, prediction_display],
        )

        step_btn.click(
            fn=on_step,
            inputs=[action_dropdown],
            outputs=outputs_step,
            show_progress="hidden",
        )
        reset_btn.click(
            fn=on_reset,
            inputs=[],
            outputs=outputs_step,
            show_progress="hidden",
        )

        def toggle_play(
            is_playing: bool, speed: float
        ) -> tuple[bool, gr.Timer, gr.Button]:
            now_playing = not is_playing
            return (
                now_playing,
                gr.Timer(value=1.0 / max(speed, 0.1), active=now_playing),
                gr.Button(value=PAUSE_LABEL if now_playing else PLAY_LABEL),
            )

        def change_speed(is_playing: bool, speed: float) -> gr.Timer:
            return gr.Timer(value=1.0 / max(speed, 0.1), active=is_playing)

        play_btn.click(
            fn=toggle_play,
            inputs=[playing, fps_slider],
            outputs=[playing, timer, play_btn],
            show_progress="hidden",
        )
        fps_slider.change(
            fn=change_speed,
            inputs=[playing, fps_slider],
            outputs=[timer],
            show_progress="hidden",
        )
        timer.tick(
            fn=on_step,
            inputs=[action_dropdown],
            outputs=outputs_step,
            show_progress="hidden",
        )

    return app


def main() -> None:
    app = build_app()
    app.launch(
        theme=build_theme(),
        css=CUSTOM_CSS,
        js=KEYBOARD_JS,
    )
