# Ejercicio Práctico 31/05/2025

## 1. Creación de Instancia en Azure OpenAI/Gemini/OpenAI

## 2. Creación de Bot en telegram (telegram instalado)

### 2.1 Obtención de id de telegram (!= @my_user)
[Follow this guide](https://www.securitylab.lat/news/554391.php)

## 3. Descarga de project base

.vscode/launch.json

{
"configurations": [
{
"name": "Main telegram bot",
"type": "python",
"request": "launch",
"program": "tu_fichero.py",
"console": "integratedTerminal"
},]}

## 4. Separación por equipos y diseño de la solución. 

.env template 

```
export OPENAI_API_KEY = x
export TELEGRAM_BOT_TOKEN = x
export ALLOWED_USERS = [123343423, 34435435656, 654646554, 565465546]
```

Inclusión de dichos usuarios en el código: 
```python
ast.literal_eval(os.getenv("ALLOWED_USERS"))
```


### 4.1 Creación whispers endpoint en Azure

## 5. Presentación de PRs, mergeo y resultados obtenidos. 