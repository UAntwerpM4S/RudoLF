# Verantwoording Workflow

## Doel van deze workflow

Deze notitie beschrijft de huidige werking van de framework-entrypoints in combinatie met de AirSim-integratie. Het doel van deze opzet is om:

- de klassieke frameworkflow te behouden;
- AirSim-specifieke logica apart en beheersbaar te houden;
- framework-keuzes en AirSim-keuzes van elkaar te scheiden;
- graceful fallback te voorzien wanneer UE5 niet beschikbaar is.

De huidige structuur is dus bewust opgesplitst in een generieke entrypoint en een AirSim-uitbreiding.

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

- AirSim training;
- Fossen-gebaseerde visualisatie in UE5;
- inference-gebaseerde visualisatie in UE5.

Deze file bevat dus de gespecialiseerde AirSim-logica, terwijl `fw.main` de algemene framework-entrypoint blijft.

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

- `model_visualise`;
- `inference_visualise`;
- `training.speed_profile` (`normal`, `safe`, `fast`, `turbo`, `custom`);
- `training.simulation_speed` en `training.frames_per_step` (bij `custom`);
- `save_slot` en andere UE5/AirSim-instellingen.

Omdat dit bestand geldige JSON moet blijven, worden toelichtingen niet als comments geschreven maar in `_remarks`-velden.

Deze beslissingen horen logisch thuis in de AirSim-entrypoint en niet in de klassieke generieke `main`.

### 3. Graceful fallback bij ontbrekende UE5-verbinding

AirSim-visualisatie vereist een actieve UE5/AirSim-verbinding. Wanneer UE5 niet draait of niet bereikbaar is, mag de volledige frameworkrun niet onnodig crashen.

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
  - modulaire framework-modelvisualisatie in UE5;
  - inference-visualisatie in UE5.

Met andere woorden: bij AirSim wordt `visualize` niet meer gekozen via een `main`-blok in runtime settings, maar via de AirSim-specifieke visualisatievlaggen.

## Gedrag met en zonder UE5

### UE5 draait wel

Als UE5 bereikbaar is:

- AirSim-visualisatie werkt;
- AirSim-training kan starten;
- `fw.main` zal bij AirSim-selectie correct delegeren en daar blijven.

### UE5 draait niet

Als UE5 niet bereikbaar is:

- een UE5-afhankelijke AirSim-visualisatie zal netjes stoppen met melding;
- `fw.main_with_airsim` zal dan eindigen zonder traceback;
- `fw.main` zal in dat geval terugvallen op zijn klassieke frameworkflow.

Dit gedrag is bewust gekozen zodat de generieke framework-entrypoint bruikbaar blijft, ook wanneer de AirSim-omgeving tijdelijk niet beschikbaar is.

## Waarom niet alles in één `main`

Een volledig samengevoegde `main` zou technisch mogelijk zijn, maar is minder duidelijk qua verantwoordelijkheid.

De huidige opzet maakt expliciet onderscheid tussen:

- algemene frameworklogica;
- AirSim-specialisatie;
- runtime-gestuurde visualisatieroutes binnen de AirSim-tak;
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

## Samenvatting

De huidige workflow is dus bewust opgebouwd rond een generieke basis-entrypoint en een gespecialiseerde AirSim-uitbreiding.

Belangrijkste redenen:

- behoud van de klassieke frameworkflow;
- duidelijke scheiding van verantwoordelijkheden;
- gescheiden framework- en AirSim-configuratie;
- graceful fallback wanneer UE5 niet beschikbaar is.

Daardoor is de huidige integratie robuuster, beter uitlegbaar en praktischer in gebruik tijdens ontwikkeling en testen.
