# Verantwoording Workflow

## Doel van deze workflow

Deze notitie beschrijft de huidige werking van de framework-entrypoints in combinatie met de AirSim-integratie. Het doel van deze opzet is om:

- de klassieke frameworkflow te behouden;
- AirSim-specifieke logica apart en beheersbaar te houden;
- framework-keuzes en AirSim-keuzes van elkaar te scheiden;
- graceful fallback te voorzien wanneer UE5 niet beschikbaar is.

De huidige structuur is bewust opgesplitst in een generieke entrypoint en een AirSim-uitbreiding.

## Huidige bestanden en rollen

### `fw.main`

`fw.main` is de standaard framework-entrypoint.

Deze file doet drie dingen:

1. De framework-config lezen (`env_type` en `action`).
2. Controleren of `env_type = AirSimEnv` en dan eventueel delegeren naar `fw.main_with_airsim`.
3. Indien AirSim niet geselecteerd is, of indien de AirSim-flow faalt, de klassieke frameworkflow uitvoeren.

De klassieke frameworkflow gebruikt de waarden uit `fw.config.CONFIG_PARAMETERS`:

- `env_type` bepaalt welke framework-omgeving gekozen wordt;
- `action` bepaalt of getraind of gevisualiseerd wordt.

### `fw.main_with_airsim`

`fw.main_with_airsim` is de AirSim-uitbreiding van `fw.main`.

Deze entrypoint leest de AirSim-runtime-instellingen en beslist welke AirSim-flow uitgevoerd moet worden:

- AirSim training (met geconfigureerd snelheidsprofiel);
- Framework-gebaseerde visualisatie in UE5 (`run_visualise_models.py`);
- Inference-gebaseerde visualisatie in UE5 (`run_inference.py`).

Deze file bevat de gespecialiseerde AirSim-logica, terwijl `fw.main` de algemene framework-entrypoint blijft.

## Waarom deze opsplitsing

De opsplitsing is gekozen om drie redenen.

### 1. Scheiding tussen generieke en AirSim-specifieke logica

De gewone frameworklogica en de AirSim-logica hebben andere verantwoordelijkheden.

- `fw.main` moet bruikbaar blijven voor gewone framework-runs.
- `fw.main_with_airsim` moet AirSim-gedrag kunnen uitvoeren dat afhangt van UE5, CosysAirSim en AirSim-runtime-opties.

Door deze twee te scheiden blijft de basis-entrypoint leesbaar en blijft de AirSim-uitbreiding uitbreidbaar.

### 2. Gescheiden configuratielagen

De generieke frameworkflow en de AirSim-flow gebruiken niet langer dezelfde selector.

De framework-config in `fw.config` bepaalt:

- `env_type`: bijvoorbeeld `PySimEnv`, `FhSimEnv`, `PySimPolarEnv`, `LunarLander` of `AirSimEnv`;
- `action`: `train` of `visualize`.

De AirSim-runtime-instellingen in `airsim/config/runtime_settings.json` bepalen onder meer:

- `model_visualise` en `inference_visualise` — welke visualisatieroute gevolgd wordt;
- `training.speed_profile` — `normal`, `safe`, `fast`, `turbo` of `custom`;
- `training.simulation_speed` en `training.frames_per_step` (bij `custom`);
- `training.curriculum_enabled` — zachte curriculumondersteuning via offtrack-limietdaling;
- `training.steering_dynamics_enabled` — low-pass filter + snelheidslimiet op stuuracties;
- `training.normalize_reward` en `training.reward_normalization_scale`;
- `save_slot` en andere UE5/AirSim-instellingen.

Omdat dit bestand geldige JSON moet blijven, worden toelichtingen niet als comments geschreven maar in `_remarks`-velden.

Deze beslissingen horen logisch thuis in de AirSim-entrypoint en niet in de klassieke generieke `main`.

### 3. Graceful fallback bij ontbrekende UE5-verbinding

AirSim-visualisatie en -training vereisen een actieve UE5/AirSim-verbinding. Wanneer UE5 niet draait of niet bereikbaar is, mag de volledige frameworkrun niet onnodig crashen.

Daarom is de huidige flow als volgt:

- `fw.main` probeert eerst de AirSim-uitbreiding indien `env_type = AirSimEnv` dat vraagt;
- als de AirSim-uitbreiding correct slaagt, stopt de flow daar;
- als de AirSim-uitbreiding netjes faalt door ontbrekende UE5-connectie, valt `fw.main` terug op de standaard frameworkflow.

## Concreet verloop van de workflow

## Scenario 1: `python -m fw.main`

Wanneer `fw.main` gestart wordt, gebeurt dit:

1. Framework-config wordt gelezen.
2. `env_type` wordt opgelost naar een canonieke omgevingsnaam.
3. Indien `env_type = AirSimEnv`, wordt gedelegeerd naar `fw.main_with_airsim`.
4. Indien die AirSim-flow succesvol is, stopt `fw.main`.
5. Indien die AirSim-flow met foutcode eindigt, wordt teruggevallen op de standaard frameworkflow.
6. Indien `env_type` geen AirSim is, blijft `fw.main` in de generieke frameworkflow.

De standaard frameworkflow gebruikt momenteel standaard:

- `env_type = "PySimEnv"`
- `action = "train"`

maar beide kunnen in `fw.config` aangepast worden.

Dat betekent dat de fallback standaard resulteert in een klassieke PySim-trainingsrun, maar desgewenst ook een andere framework-omgeving of visualisatie kan zijn.

## Scenario 2: `python -m fw.main_with_airsim`

Wanneer `fw.main_with_airsim` rechtstreeks gestart wordt, wordt geen fallback uitgevoerd.

Deze entrypoint gaat ervan uit dat de gebruiker bewust de AirSim-flow wil forceren.

Dat betekent:

- als UE5 draait, wordt de gevraagde AirSim-flow uitgevoerd;
- als UE5 niet draait en de gekozen flow UE5 nodig heeft, stopt het proces met een nette foutmelding.

## Gedrag voor `train` en `visualize`

### In `fw.main`

Binnen de standaard frameworkflow geldt:

- `train` betekent trainen van de omgevingen in `envs`;
- `visualize` betekent `trainer.visualize(...)` uitvoeren op de omgevingen in `envs`.

Omdat `envs` uit `env_type` afgeleid wordt, betekent de generieke flow standaard:

- trainen van `PySimEnv`, of
- visualiseren van `PySimEnv`.

Als de gebruiker in `fw.config` een andere omgeving kiest, geldt hetzelfde patroon voor die omgeving.

### In `fw.main_with_airsim`

Binnen de AirSim-flow geldt:

- `train` met `AirSimEnv` start training in AirSim;
- trainingssnelheid en stepping-gedrag worden gestuurd via `training.speed_profile` of via expliciete `training.*` waarden;
- AirSim-visualisatie kiest op basis van runtime-instellingen tussen:
  - modulaire framework-modelvisualisatie in UE5 (`model_visualise`);
  - inference-visualisatie in UE5 (`inference_visualise`).

Met andere woorden: bij AirSim wordt `visualize` niet meer gekozen via een `main`-blok in runtime settings, maar via de AirSim-specifieke visualisatievlaggen.

## Trainingssnelheidsprofielen

De trainingssnelheid wordt bepaald door `training.speed_profile` in `runtime_settings.json`.

| Profiel | Simsnelheid | Pause stepping | time_step | Gebruik |
|---------|-------------|----------------|-----------|---------|
| `normal` | 1.0× | uit | 0.1 s | Debug / realtime |
| `safe` | 20.0× | aan | 0.05 s | **Aanbevolen voor nieuwe training** |
| `fast` | 40.0× | aan | 0.1 s | Snellere iteratie |
| `turbo` | 80.0× | aan | 0.1 s | Maximum doorvoer |
| `custom` | vrij | vrij | vrij | Handmatig |

`safe` is aanbevolen voor initiële trainingsruns. Het combineert deterministisch pause-stepping met een conservatieve simsnelheid om fysische instabiliteit te reduceren.

## Curriculumleren

Curriculumleren wordt gedeeltelijk ondersteund via een zachte offtrack-limietdaling in `AirSimEnv`.

Wanneer `curriculum_enabled=true`:

- de offtrack-limiet start ruim en daalt geleidelijk over ca. 450 episodes;
- dit laat de policy eerst globale routenavigatie leren, waarna strakker sporen vereist is.

`TrainingLogSummary` in `train/training_log_analysis.py` analyseert episode-CSV-logs en stelt een curriculumfrontier voor op basis van mediaan/modusvoortgang per checkpoint en het aandeel high-speed offtrack-runs.

**Volledig curriculumschema en optioneel transfer learning zijn nog niet actief. Dit is het laatste geplande implementatiestadium.**

## Vroegtijdig stoppen

`StrictSuccessStoppingCallback` stopt training automatisch wanneer de policy een configureerbaar aantal opeenvolgende strict-success-episodes bereikt.

Succescriteria per episode:

1. Alle routecheckpoints gepasseerd.
2. Maximale CTE ≤ `strict_cte_limit_m` (standaard 10 m).
3. Gemiddeld episoderendement ≥ `min_average_episode_reward` (standaard 0.95 genormaliseerd).

## Stuurgedrag en steering dynamics

Wanneer `steering_dynamics_enabled=true` worden policy-outputs niet ruw doorgestuurd naar UE5.

Twee filters worden opeenvolgend toegepast:

1. **Low-pass filter** — exponentiële afvlakking met `steering_smoothing_alpha`.
2. **Snelheidslimiet** — de verandering per stap wordt beperkt tot `max_steering_rate_per_second × time_step`.

Dit emuleert de traagheid van scheepsbesturing en voorkomt dat discrete actiessprongen de fysica destabiliseren.

## Gedrag met en zonder UE5

### UE5 draait wel

Als UE5 bereikbaar is:

- AirSim-visualisatie werkt;
- AirSim-training kan starten;
- `fw.main` zal bij AirSim-selectie correct delegeren en daar blijven.

### UE5 draait niet

Als UE5 niet bereikbaar is:

- een UE5-afhankelijke AirSim-visualisatie of -training zal netjes stoppen met melding;
- `fw.main_with_airsim` zal dan eindigen zonder traceback;
- `fw.main` zal in dat geval terugvallen op zijn klassieke frameworkflow.

Dit gedrag is bewust gekozen zodat de generieke framework-entrypoint bruikbaar blijft, ook wanneer de AirSim-omgeving tijdelijk niet beschikbaar is.

## Waarom niet alles in één `main`

Een volledig samengevoegde `main` zou technisch mogelijk zijn, maar is minder duidelijk qua verantwoordelijkheid.

De huidige opzet maakt expliciet onderscheid tussen:

- algemene frameworklogica;
- AirSim-specialisatie;
- runtime-gestuurde visualisatieroutes binnen de AirSim-tak;
- trainingssnelheidsprofielen en deterministische steppingmodi;
- fallback-gedrag.

Daardoor is het eenvoudiger om:

- AirSim-functionaliteit verder uit te breiden;
- fouten in UE5-connectiviteit op te vangen;
- de klassieke frameworkflow intact te houden.

## Huidige praktische afspraak

De aanbevolen standaard entrypoint is momenteel:

```bash
python -m fw.main
```

Deze entrypoint:

- probeert automatisch de AirSim-flow als `env_type = AirSimEnv` dat vraagt;
- gebruikt voor de generieke frameworkflow de waarden `env_type` en `action` uit `fw.config`;
- valt terug op de framework-defaults indien nodig.

De AirSim-entrypoint:

```bash
python -m fw.main_with_airsim
```

wordt gebruikt wanneer expliciet een pure AirSim-run gewenst is, zonder fallback naar de generieke frameworkflow.

## Geplande uitbreidingen

Het volgende is gepland als laatste implementatiestadium:

- **Volledig curriculumschema** — gestructureerde multi-fase curriculumprogressie op basis van `TrainingLogSummary`-aanbevelingen.
- **Optioneel transfer learning** — hergebruik van policy-gewichten getraind in `PySimEnv` of `FhSimEnv` als startpunt voor `AirSimEnv`-training, om initieel leren te versnellen.

Deze functies zijn bewust uitgesteld totdat de basistraining in `AirSimEnv` stabiel en reproduceerbaar is.

## Samenvatting

De huidige workflow is bewust opgebouwd rond een generieke basis-entrypoint en een gespecialiseerde AirSim-uitbreiding.

Belangrijkste redenen:

- behoud van de klassieke frameworkflow;
- duidelijke scheiding van verantwoordelijkheden;
- gescheiden framework- en AirSim-configuratie;
- meerdere trainingssnelheidsprofielen met deterministische stepping;
- zachte curriculumondersteuning;
- graceful fallback wanneer UE5 niet beschikbaar is.

Daardoor is de huidige integratie robuuster, beter uitlegbaar en praktischer in gebruik tijdens ontwikkeling en testen.
