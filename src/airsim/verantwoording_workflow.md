# Verantwoording Workflow

## Doel van deze workflow

Deze notitie beschrijft de huidige werking van de framework-entrypoints in combinatie met de AirSim-integratie. Het doel van deze opzet is om:

- de klassieke frameworkflow te behouden;
- AirSim-specifieke logica apart en beheersbaar te houden;
- runtime-gestuurde AirSim-keuzes mogelijk te maken;
- graceful fallback te voorzien wanneer UE5 niet beschikbaar is.

De huidige structuur is dus bewust opgesplitst in een generieke entrypoint en een AirSim-uitbreiding.

## Huidige bestanden en rollen

### `fw.main`

`fw.main` is de standaard framework-entrypoint.

Deze file doet twee dingen:

1. Eerst controleren of de runtime-instellingen expliciet AirSim selecteren.
2. Indien AirSim niet geselecteerd is, of indien de AirSim-flow faalt, de klassieke frameworkflow uitvoeren.

De klassieke frameworkflow is momenteel nog hardcoded en gebruikt standaard `PySimEnv`.

### `fw.main_with_airsim`

`fw.main_with_airsim` is de AirSim-uitbreiding van `fw.main`.

Deze entrypoint leest de runtime-instellingen en beslist welke AirSim-flow uitgevoerd moet worden:

- AirSim training;
- Fossen-gebaseerde visualisatie in UE5;
- inference-gebaseerde visualisatie in UE5.

Deze file bevat dus de gespecialiseerde AirSim-logica, terwijl `fw.main` de algemene framework-entrypoint blijft.

## Waarom deze opsplitsing

De opsplitsing is gekozen om drie redenen.

### 1. Scheiding tussen generieke en AirSim-specifieke logica

De gewone frameworklogica en de AirSim-logica hebben andere verantwoordelijkheden.

- `fw.main` moet bruikbaar blijven voor gewone framework-runs.
- `fw.main_with_airsim` moet runtime-gestuurd AirSim-gedrag kunnen uitvoeren.

Door deze twee te scheiden blijft de basis-entrypoint leesbaar en blijft de AirSim-uitbreiding uitbreidbaar.

### 2. Runtime-gestuurde AirSim-keuze

AirSim-visualisatie en AirSim-training hangen af van runtime-instellingen in `airsim/config/runtime_settings.json`.

Die instellingen bepalen onder meer:

- `action`: `train` of `visualize`;
- `train_environment`: bijvoorbeeld `AirSimEnv`;
- `fossen_visualise`;
- `inference_visualise`.

Deze beslissingen horen logisch thuis in de AirSim-entrypoint en niet in de klassieke generieke `main`.

### 3. Graceful fallback bij ontbrekende UE5-verbinding

AirSim-visualisatie vereist een actieve UE5/AirSim-verbinding. Wanneer UE5 niet draait of niet bereikbaar is, mag de volledige frameworkrun niet onnodig crashen.

Daarom is de huidige flow als volgt:

- `fw.main` probeert eerst de AirSim-uitbreiding indien runtime dat vraagt;
- als de AirSim-uitbreiding correct slaagt, stopt de flow daar;
- als de AirSim-uitbreiding netjes faalt door ontbrekende UE5-connectie, valt `fw.main` terug op de standaard frameworkflow.

## Concreet verloop van de workflow

## Scenario 1: `python -m fw.main`

Wanneer `fw.main` gestart wordt, gebeurt dit:

1. Runtime-instellingen worden gelezen.
2. Indien `AirSimEnv` geselecteerd is met actie `train` of `visualize`, wordt gedelegeerd naar `fw.main_with_airsim`.
3. Indien die AirSim-flow succesvol is, stopt `fw.main`.
4. Indien die AirSim-flow met foutcode eindigt, wordt teruggevallen op de standaard frameworkflow.

De standaard frameworkflow gebruikt momenteel standaard:

- `action = "train"`
- `envs = [PySimEnv]`

Dat betekent dat de fallback vandaag resulteert in een klassieke PySim-trainingsrun.

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

Omdat `envs` momenteel hardcoded `PySimEnv` bevat, betekent de generieke flow dus standaard:

- trainen van `PySimEnv`, of
- visualiseren van `PySimEnv`.

### In `fw.main_with_airsim`

Binnen de AirSim-flow geldt:

- `train` met `AirSimEnv` start training in AirSim;
- `visualize` met `AirSimEnv` kiest op basis van runtime-instellingen tussen:
  - Fossen-visualisatie in UE5;
  - inference-visualisatie in UE5.

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
- runtime-gestuurde visualisatieroutes;
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

- probeert automatisch de AirSim-flow als runtime dat vraagt;
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
- runtime-gestuurde AirSim-keuzes;
- graceful fallback wanneer UE5 niet beschikbaar is.

Daardoor is de huidige integratie robuuster, beter uitlegbaar en praktischer in gebruik tijdens ontwikkeling en testen.