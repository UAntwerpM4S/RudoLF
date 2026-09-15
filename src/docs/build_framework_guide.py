from __future__ import annotations

import html
import zipfile
from pathlib import Path


OUTPUT = Path("docs/RudoLF_src_framework_guide.docx")


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def paragraph(text: str = "", style: str | None = None) -> str:
    ppr = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>" if style else ""
    return f"<w:p>{ppr}<w:r><w:t>{esc(text)}</w:t></w:r></w:p>"


def bullet(text: str) -> str:
    return (
        "<w:p>"
        "<w:pPr><w:pStyle w:val=\"Bullet\"/>"
        "<w:ind w:left=\"540\" w:hanging=\"270\"/></w:pPr>"
        f"<w:r><w:t>{esc('• ' + text)}</w:t></w:r>"
        "</w:p>"
    )


def code_line(text: str) -> str:
    return (
        "<w:p><w:pPr><w:pStyle w:val=\"Code\"/></w:pPr>"
        f"<w:r><w:t xml:space=\"preserve\">{esc(text)}</w:t></w:r></w:p>"
    )


def table(headers: list[str], rows: list[list[str]]) -> str:
    col_count = len(headers)
    col_width = int(9360 / col_count)
    grid = "".join(f"<w:gridCol w:w=\"{col_width}\"/>" for _ in headers)

    def cell(text: str, header: bool = False) -> str:
        fill = "<w:shd w:fill=\"E8EEF5\"/>" if header else ""
        bold_start = "<w:b/>" if header else ""
        return (
            "<w:tc>"
            f"<w:tcPr><w:tcW w:w=\"{col_width}\" w:type=\"dxa\"/>{fill}</w:tcPr>"
            f"<w:p><w:r><w:rPr>{bold_start}</w:rPr><w:t>{esc(text)}</w:t></w:r></w:p>"
            "</w:tc>"
        )

    header_row = "<w:tr>" + "".join(cell(h, True) for h in headers) + "</w:tr>"
    body_rows = "".join("<w:tr>" + "".join(cell(c) for c in row) + "</w:tr>" for row in rows)
    return (
        "<w:tbl>"
        "<w:tblPr><w:tblW w:w=\"9360\" w:type=\"dxa\"/>"
        "<w:tblInd w:w=\"120\" w:type=\"dxa\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:color=\"B8C2CC\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:color=\"B8C2CC\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:color=\"B8C2CC\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:color=\"B8C2CC\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:color=\"D0D7DE\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:color=\"D0D7DE\"/>"
        "</w:tblBorders>"
        "<w:tblCellMar><w:top w:w=\"80\" w:type=\"dxa\"/><w:bottom w:w=\"80\" w:type=\"dxa\"/>"
        "<w:start w:w=\"120\" w:type=\"dxa\"/><w:end w:w=\"120\" w:type=\"dxa\"/></w:tblCellMar>"
        "</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        f"{header_row}{body_rows}"
        "</w:tbl>"
    )


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:pPr><w:spacing w:after="120" w:line="300" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:pPr><w:spacing w:after="200"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:b/><w:color w:val="0B2545"/><w:sz w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:pPr><w:spacing w:after="200"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:color w:val="555555"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="Heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
    <w:pPr><w:spacing w:before="360" w:after="200"/><w:outlineLvl w:val="0"/></w:pPr>
    <w:rPr><w:b/><w:color w:val="2E74B5"/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="Heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
    <w:pPr><w:spacing w:before="280" w:after="140"/><w:outlineLvl w:val="1"/></w:pPr>
    <w:rPr><w:b/><w:color w:val="2E74B5"/><w:sz w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="Heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
    <w:pPr><w:spacing w:before="200" w:after="100"/><w:outlineLvl w:val="2"/></w:pPr>
    <w:rPr><w:b/><w:color w:val="1F4D78"/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Bullet">
    <w:name w:val="Bullet"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:after="80" w:line="300" w:lineRule="auto"/></w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Code">
    <w:name w:val="Code"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:after="40" w:line="240" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="Microsoft YaHei"/><w:sz w:val="18"/></w:rPr>
  </w:style>
</w:styles>"""


def document_xml() -> str:
    blocks: list[str] = []
    add = blocks.append

    add(paragraph("RudoLF src Framework Guide", "Title"))
    add(paragraph("Structure, main.py execution flow, function responsibilities, and extension points for classical path-following control.", "Subtitle"))

    add(paragraph("1. Executive Overview", "Heading1"))
    add(paragraph("This document explains the current src framework from the perspective of someone who wants to keep a new classical path-following controller consistent with the existing codebase. The key idea is that fw/main.py is an orchestration entry point. It reads config.py, applies profiles, builds environments through env_factory.py, delegates training or visualization to Trainer, and bundles run outputs. The actual vessel physics, observation construction, reward calculation, policy learning, and plotting happen in downstream modules."))
    add(paragraph("A new controller should therefore avoid duplicating framework setup logic. It should read the same config.py values, use the same ship factory and simulator abstractions, write outputs in the same run-archive style, and be exposed through main.py as a normal action."))

    add(paragraph("2. Top-Level Directory Structure", "Heading1"))
    add(table(["Path", "Role"], [
        ["fw/main.py", "Main orchestration entry point for train, visualize, evaluate_reward, and the new path_following action."],
        ["fw/configuration/", "Configuration, profiles, environment factory, and stop conditions."],
        ["fw/environments/", "Gymnasium-compatible simulation environments, including PySimEnv and COLAV variants."],
        ["fw/simulators/", "Physical simulator wrappers, including ShipSimulator and FH/simSync bridges."],
        ["fw/ships/", "Ship specifications, ship factory, actuator mapping, actuator dynamics, and vessel state containers."],
        ["fw/dynamics/", "Dynamics models: Fossen, Nomoto, UGent surrogate, UAntwerp surrogate, and FH tabular bridge."],
        ["fw/policies/", "PPO-based learning agents and policy/network implementations."],
        ["fw/scheduling/", "Trainer, scheduler, and agent-manager layers that connect environments to policies."],
        ["fw/postprocessing/", "Episode collection, metrics computation, Matplotlib/Plotly rendering, and animation export."],
        ["fw/control/", "Classical-control experiments, including the LOS/PID path-following baseline."],
        ["fw/reward/", "Reward component calculation for RL environments."],
        ["fw/collision_avoidance/", "COLREG/COLAV scenario entities, classifiers, and statistics."],
        ["fw/surrogate/", "Data generation, training, and inference utilities for surrogate dynamics models."],
        ["fw/common/", "Shared constants, path utilities, logging, bundling, AirSim helpers, and support classes."],
    ]))

    add(paragraph("3. main.py Runtime Flow", "Heading1"))
    add(paragraph("The main.py file is intentionally script-like. Most work happens inside the __main__/fw.main guard so environment variables are set before importing heavy numerical libraries."))
    for line in [
        "Set OMP_NUM_THREADS and MKL_NUM_THREADS before NumPy/Torch imports.",
        "Load config.py through get_config_parameters(), which applies the selected profile once.",
        "Normalize the action field: train, visualize, evaluate_reward, or path_following.",
        "Prepare logging with a readable environment label.",
        "Resolve env_type into canonical environment names.",
        "Select the active VesselProfile for later object-level side effects.",
        "For path_following, delegate directly to fw.control.path_following_los_pid.main().",
        "For train/visualize, initialize environment registration, build environment wrappers, and execute standard batches.",
        "For evaluate_reward, build isolated environments for each reward configuration and export comparison results.",
        "Bundle logs, config.py, resolved configuration, and policy artifacts for traceability.",
    ]:
        add(bullet(line))

    add(paragraph("4. Key main.py Functions", "Heading1"))
    add(table(["Function", "Responsibility", "Important Downstream Calls"], [
        ["_run_airsim_extension(selected_action)", "Runs the AirSim-specific entrypoint and falls back gracefully if UE5/CosysAirSim is unavailable.", "runpy.run_module('fw.main_with_airsim')"],
        ["_select_agent_type(envs)", "Chooses PPO, MPPO, or MPSA after environments are built.", "Checks include_scenario_one_hot and MultiSimEnv training_name."],
        ["_run_standard_batch(envs)", "Applies profile side effects, creates Trainer, runs train or visualize, and bundles outputs.", "apply_profile(), Trainer(), bundle_run_output()"],
        ["evaluate_reward branch", "Trains/evaluates several reward configurations in isolated runs.", "build_default_environment(), Trainer.evaluate_reward(), export_evaluation_results()"],
        ["environment loop", "Preserves ordered env_type execution and separates AirSim from regular framework batches.", "build_default_environment(), _run_standard_batch(), _run_airsim_extension()"],
    ]))

    add(paragraph("5. Configuration Flow", "Heading1"))
    add(paragraph("The framework separates raw configuration from resolved runtime configuration. CONFIG_PARAMETERS in fw/configuration/config.py is the editable source. get_config_parameters() deep-copies it, looks up the selected profile, validates the profile, applies profile-level overrides, and returns the resolved dict."))
    add(code_line("CONFIG_PARAMETERS -> get_config_parameters() -> apply_profile(profile, config=...) -> framework_config"))
    add(paragraph("Profile application has several layers:"))
    add(bullet("ConfigApplier changes ship_model and dynamics_model in the config dict."))
    add(bullet("EnvApplier changes live environment fields such as initial_speed, enable_smoothing, and rudder-angle scaling."))
    add(bullet("SimulatorApplier changes actuator limits inside the simulator's mutable ShipSpecifications."))
    add(bullet("ActionAdapter can transform raw [rudder, thrust] actions before environment stepping."))

    add(paragraph("6. Environment Factory and Environment Construction", "Heading1"))
    add(paragraph("fw/configuration/env_factory.py maps user-facing env_type selectors to registered wrapper classes. Each wrapper receives a scoped config dict containing only the keys it needs. This prevents unrelated config fields from leaking into environments."))
    add(table(["Builder / Wrapper", "Purpose"], [
        ["resolve_environment_name(s)", "Converts aliases such as pysim or fhsim into canonical registry names."],
        ["build_default_environment()", "Chooses the correct scoped config builder and returns a wrapper instance."],
        ["PySimEnvironment", "Builds PySimEnv with map files, start/target positions, ship model, dynamics model, wind/current, reward settings."],
        ["FhSimEnvironment", "Constructs an FH bridge and then drives PySimEnv with Model.FH_TABULAR."],
        ["AirSimEnvironment", "Builds the UE5/CosysAirSim-backed environment and optional action-repeat wrapper."],
        ["MultiPySimEnvironment", "Creates two PySimEnv instances and wraps them in MultiSimEnv."],
        ["PySimColavEnvironment", "Builds the collision-avoidance environment variant."],
    ]))

    add(paragraph("7. PySimEnv Responsibilities", "Heading1"))
    add(paragraph("PySimEnv is the main Python-side ship navigation environment. It loads the Western Scheldt map and trajectory CSVs, trims the route to the configured start and target positions, builds checkpoints, creates the ship and simulator, defines normalized observation/action spaces, computes reward, and handles rendering."))
    add(code_line("main.py -> env_factory.PySimEnvironment -> PySimEnv(...) -> create_ship() -> create_simulator() -> ShipSimulator"))
    add(paragraph("Important PySimEnv operations:"))
    add(bullet("_load_environment_data() loads overall/depth/trajectory CSVs and converts the route into checkpoints."))
    add(bullet("create_simulator() chooses Fossen, Nomoto, UGent, UAntwerp, or FH_TABULAR dynamics and wraps it in ShipSimulator."))
    add(bullet("_initialize_simulation() sets initial position, heading, speed, checkpoint index, and simulator state."))
    add(bullet("_get_obs() returns normalized velocities, distance-to-checkpoint, cross-track error, heading errors, previous action, and LOS heading error."))
    add(bullet("step(action) accepts normalized [rudder, thrust], advances the physical simulator, computes reward, and returns Gymnasium-style outputs."))

    add(paragraph("8. Ship, Actuator, and Dynamics Layers", "Heading1"))
    add(table(["Layer", "Key Files", "Role"], [
        ["Ship model", "fw/ships/ship_factory.py, ship_specs.py", "Selects a ShipModel and creates ShipSpecifications with length, mass, rudder limits, thrust limits, and hydrodynamic coefficients."],
        ["Action mapping", "fw/ships/action_mapper.py", "Maps normalized action[0] to physical rudder angle and action[1] to thrust range."],
        ["Actuator dynamics", "fw/ships/actuator_model.py", "Optionally applies rudder/thrust rate limits, jitter thresholds, and saturation."],
        ["Simulator", "fw/simulators/ship_simulator.py", "Substeps dynamics at 0.1 s, applies environmental accelerations, clips velocities, and integrates x/y/heading."],
        ["Dynamics", "fw/dynamics/*.py", "Computes body-fixed surge/sway/yaw accelerations or proxies an external/surrogate model."],
    ]))
    add(paragraph("The normalized action convention is consistent across RL and classical control: controllers or policies output two values in [-1, 1]. ActionMapper converts these to physical actuator commands. Your LOS/PID controller keeps physical rudder/thrust internally and then converts back to this normalized action interface before calling ShipSimulator.step()."))

    add(paragraph("9. Training and Visualization Flow", "Heading1"))
    add(code_line("main.py -> Trainer -> AgentManager -> SimpleScheduler -> BaseAgent.learn()"))
    add(code_line("visualize: Trainer.visualize() -> AgentManager.visualize_trained_agent() -> EpisodeCollector -> MetricsComputer -> MatplotlibRenderer/AnimationRenderer"))
    add(paragraph("Trainer constructs the selected AgentManager and dynamically loads SimpleScheduler. SimpleScheduler switches through configured environments, trains the agent, saves policies, and writes the active profile next to the policy. AgentManager owns the actual PPO/MPPO/MPSA object and handles policy reuse. Visualization runs a trained policy in evaluation mode, collects episode data, computes metrics, and renders plots."))

    add(paragraph("10. Classical LOS/PID Path-Following Integration", "Heading1"))
    add(paragraph("The new fw/control/path_following.py script follows the framework style but remains independent from RL training. It reads config.py through get_config_parameters(), builds the configured ship and dynamics model, loads the same map CSVs, runs LOS guidance plus heading/surge PID control, and writes a timestamped run archive."))
    add(table(["Feature", "Implementation"], [
        ["Config source", "get_config_parameters(), same as main.py."],
        ["Ship/dynamics selection", "create_ship(config.ship_model) and build_dynamics_model(config.dynamics_model, specs)."],
        ["Initial/target position", "config.start_pos and config.target_pos."],
        ["Map and route files", "config.overall_contour, config.depth_contour, config.trajectory."],
        ["Guidance", "LOSGuidance with default lookahead = 3 * configured ship length."],
        ["Heading control", "Default PD: kp=1.5, ki=0.0, kd=20.0."],
        ["Speed control", "Surge PID from u=0 to target_speed=4 m/s."],
        ["Outputs", "simulation_data.csv, trajectory_path.csv, metrics_summary.csv/json, resolved_config.json, PNG maps, summary.png, index.html, trajectory.gif."],
    ]))
    add(paragraph("main.py now exposes this through action='path_following'. To use it, set CONFIG_PARAMETERS['action'] = 'path_following' in config.py and run python -m fw.main. The path-following script still owns its own command-line arguments, so it can also be run directly with python -m fw.control.path_following_los_pid."))

    add(paragraph("11. Practical Extension Guidance", "Heading1"))
    add(bullet("If a new controller uses the same physical simulator, keep the normalized action interface: [rudder_action, thrust_action] in [-1, 1]."))
    add(bullet("Read config.py through get_config_parameters() so profile overrides are respected exactly once."))
    add(bullet("Avoid hardcoding ship length, map files, start/target positions, or dynamics model; all already exist in the resolved config."))
    add(bullet("Write resolved_config.json beside numerical outputs so every run is reproducible."))
    add(bullet("Use the same run_archives timestamp pattern as main.py/run_bundler for experiment comparison."))
    add(bullet("When adding a new main.py action, keep main.py as a dispatcher and move domain logic into a module under fw/control, fw/scheduling, or fw/postprocessing as appropriate."))

    add(paragraph("12. Call Relationship Summary", "Heading1"))
    for line in [
        "Configuration path: config.py -> get_config_parameters() -> profile_applier.ConfigApplier -> framework_config.",
        "RL train path: main.py -> env_factory -> Trainer -> SimpleScheduler -> AgentManager -> PPO agent -> env.step().",
        "RL visualization path: main.py -> Trainer.visualize() -> AgentManager.visualize_trained_agent() -> EpisodeCollector -> MetricsComputer -> renderer.",
        "Physical simulation path: PySimEnv.step() -> ShipSimulator.step() -> ActionMapper -> ActuatorModel -> DynamicsBase.step() -> state integration.",
        "Classical-control path: main.py action='path_following' -> path_following_los_pid.main() -> LOSGuidance/PID -> ShipSimulator -> postprocessing outputs.",
    ]:
        add(bullet(line))

    add(paragraph("13. Recommended Next Development Steps", "Heading1"))
    add(bullet("Create a small controller interface so RL policies and classical controllers can share evaluation/postprocessing utilities."))
    add(bullet("Move common map loading and route reduction from PySimEnv/path_following_los_pid into a shared helper to avoid drift."))
    add(bullet("Add config.py fields for classical controller parameters once the LOS/PID design stabilizes."))
    add(bullet("Add a main.py smoke test for action='path_following' that runs with --no-gif or a short duration."))

    sect = (
        "<w:sectPr>"
        "<w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )
    body = "".join(blocks) + sect
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>{body}</w:body>
</w:document>"""


def build_docx() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/_rels/document.xml.rels", doc_rels)
        docx.writestr("word/document.xml", document_xml())
        docx.writestr("word/styles.xml", styles_xml())


if __name__ == "__main__":
    build_docx()
    print(OUTPUT.resolve())
