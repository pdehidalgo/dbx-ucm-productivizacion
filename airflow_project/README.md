

```bash
docker compose up --build

# Mucho cuidado 
docker-compose down --volumes --remove-orphans
docker system prune --all --volumes --force

docker-compose build --no-cache
docker-compose up --force-recreate

```