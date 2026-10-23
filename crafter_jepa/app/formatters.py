from html import escape

from crafter_jepa.app.env_session import EnvSnapshot
from crafter_jepa.app.model_session import StepPrediction

MSE_GOOD = 0.02
MSE_OK = 0.06
MSE_BAR_MAX = 0.10


def _signed_class(x: float) -> str:
    if x > 0:
        return "pos"
    if x < 0:
        return "neg"
    return ""


def _vital(name: str, value: int, max_value: int = 9) -> str:
    pips = "".join(
        f'<span class="pip{" on" if i < value else ""}"></span>'
        for i in range(max_value)
    )
    low = " low" if value <= 3 else ""
    return (
        f'<div class="vital{low}">'
        f'<span class="vital-name">{name}</span>'
        f'<span class="pips">{pips}</span>'
        f'<span class="vital-num">{value}</span>'
        f"</div>"
    )


def format_vitals(snap: EnvSnapshot) -> str:
    status = (
        '<span class="dot bad"></span>Koniec epizodu'
        if snap.done
        else '<span class="dot ok"></span>Aktywny'
    )
    vitals = "".join(
        [
            _vital("Zdrowie", snap.health),
            _vital("Jedzenie", snap.food),
            _vital("Woda", snap.water),
            _vital("Energia", snap.energy),
        ]
    )
    return f"""
<div class="hud">
  <div class="hud-cell">
    <span class="hud-label">Krok</span>
    <span class="hud-value">{snap.step_num}</span>
  </div>
  <div class="hud-cell">
    <span class="hud-label">Suma nagród</span>
    <span class="hud-value {_signed_class(snap.total_reward)}">{snap.total_reward:+.1f}</span>
  </div>
  <div class="hud-cell">
    <span class="hud-label">Nagroda r<sub>t</sub></span>
    <span class="hud-value {_signed_class(snap.reward)}">{snap.reward:+.1f}</span>
  </div>
  <div class="hud-cell">
    <span class="hud-label">Status</span>
    <span class="hud-value small">{status}</span>
  </div>
  <div class="hud-cell grow">
    <span class="hud-label">Parametry życiowe</span>
    <div class="vitals">{vitals}</div>
  </div>
</div>
"""


def format_model_status(algo: str, loaded_ckpt: str) -> str:
    random_init = loaded_ckpt.startswith("Inicjalizacja")
    dot = "warn" if random_init else "ok"
    detail = "losowe wagi" if random_init else escape(loaded_ckpt)
    return (
        f'<div class="model-status"><span class="dot {dot}"></span>'
        f"<span><b>{escape(algo.upper())}</b> · {detail}</span></div>"
    )


def format_decision_empty() -> str:
    return """
<div class="metric-card">
  <div class="section-title">Decyzja</div>
  <div class="empty">Brak kroku. Naciśnij <kbd>Spacja</kbd> lub uruchom odtwarzanie.</div>
</div>
"""


def format_decision(pred: StepPrediction, reward: float) -> str:
    prob_pct = pred.act_prob * 100
    return f"""
<div class="metric-card">
  <div class="section-title">Decyzja</div>
  <div class="action-name">{escape(pred.act_name)}<small>#{pred.act_idx}</small></div>
  <div class="kv">
    <div>
      <div class="kv-row">
        <span class="kv-key">Prawdopodobieństwo <i>P(a<sub>t</sub>)</i></span>
        <span class="kv-val">{prob_pct:.1f}%</span>
      </div>
      <div class="bar" style="margin-top:8px"><div style="width:{prob_pct:.1f}%"></div></div>
    </div>
    <div class="kv-row">
      <span class="kv-key">Wartość stanu <i>V(s)</i></span>
      <span class="kv-val {_signed_class(pred.v_est)}">{pred.v_est:+.3f}</span>
    </div>
    <div class="kv-row">
      <span class="kv-key">Nagroda <i>r<sub>t</sub></i></span>
      <span class="kv-val {_signed_class(reward)}">{reward:+.1f}</span>
    </div>
  </div>
</div>
"""


def format_prediction_info(pred: StepPrediction) -> str:
    header = '<div class="section-title">Przestrzeń ukryta</div>'

    if pred.algo == "ppo":
        return f"""
<div class="metric-card">
  {header}
  <div class="note">PPO uczy wyłącznie polityki i funkcji wartości — brak dekodera
  i predyktora, więc nie ma czego wizualizować.</div>
</div>
"""

    if pred.mse_val is None:
        return f"""
<div class="metric-card">
  {header}
  <div class="note">Model <b>{escape(pred.algo.upper())}</b> nie ma wytrenowanego
  predyktora ani dekodera. Uruchom <code>just train-predictor</code>.</div>
</div>
"""

    mse = pred.mse_val
    if mse < MSE_GOOD:
        dot, label = "ok", "Wysoka"
    elif mse < MSE_OK:
        dot, label = "warn", "Umiarkowana"
    else:
        dot, label = "bad", "Niska"
    bar_pct = min(mse / MSE_BAR_MAX, 1.0) * 100

    return f"""
<div class="metric-card">
  {header}
  <div>
    <div class="kv-key" style="margin-bottom:10px">MSE <i>ô(t+1)</i> vs <i>o(t+1)</i></div>
    <div class="big-num">{mse:.5f}</div>
  </div>
  <div class="bar {dot}"><div style="width:{bar_pct:.1f}%"></div></div>
  <div class="kv-row">
    <span class="kv-key">Jakość predykcji</span>
    <span class="chip"><span class="dot {dot}"></span>{label}</span>
  </div>
</div>
"""
