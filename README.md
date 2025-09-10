<p align="center"><img src="./docs/img/logo_banner.png" alt="Skypilot Catalog API" width="500"></p>

# Skypilot Catalog API

A minimal REST API that exposes instance data from SkyPilot catalog files.

## Features

- Exposes SkyPilot catalog data via a REST API
- Streams instance data
- Can filter instances by cloud, GPU availability, name, region, and quantity
- Containerized with Docker for easy deployment

## Tech Stack

- **Framework**: FastAPI
- **Server**: Uvicorn
- **Data Handling**: Pandas, Pydantic
- **Tooling**: uv, pytest, Ruff, Black, pre-commit

## Quickstart

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/skypilot-sidecar.git
    cd skypilot-sidecar
    ```

2.  **Create a virtual environment and install dependencies:**

    ```sh
    uv venv
    source .venv/bin/activate
    uv sync --locked
    ```

3.  **Configure the application:**
    Create a `.env` file and add any necessary environment variables. See the Configuration section for details.

4.  **Run the application:**

    ```sh
    uvicorn src.main:app --reload --port 8000
    ```

5.  **Access the API documentation:**
    The interactive API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Configuration

Configuration is managed via environment variables with the prefix `EXLS_SKY_`.

| Name                        | Example/Type                  | Required? | Description                                                           |
| --------------------------- | ----------------------------- | --------- | --------------------------------------------------------------------- |
| `EXLS_SKY_API_HOST`         | `0.0.0.0` (string)            | No        | Host to run the API on. Defaults to `0.0.0.0`.                        |
| `EXLS_SKY_API_PORT`         | `5555` (integer)              | No        | Port to run the API on. Defaults to `5555`.                           |
| `EXLS_SKY_CATALOG_PATH`     | `~/.sky/catalogs/v7` (string) | No        | Path to the catalog root directory. Defaults to `~/.sky/catalogs/v7`. |
| `EXLS_SKY_CATALOG_FILENAME` | `vms.csv` (string)            | No        | Name of the catalog file. Defaults to `vms.csv`.                      |

## API

The API documentation is available at `/docs` and `/redoc` when the application is running.

**Health Check:**

```sh
curl -X GET http://127.0.0.1:8000/health
```

**List Instances:**

```sh
curl -X POST http://127.0.0.1:8000/v1.0/list-instances \
-H "Content-Type: application/json" \
-d '{
  "clouds": ["aws", "gcp"],
  "gpus_only": true
}'
```

## Project Layout

```
.
├── Dockerfile
├── pyproject.toml
├── README.md
├── src
│   ├── api.py
│   ├── config.py
│   ├── main.py
│   ├── models.py
│   └── services.py
└── tests
    ├── integration
    └── unit
```

## Tests

To run the test suite, use `pytest`:

```sh
pytest
```

## Code Style

This project uses `Ruff` for linting, `Black` for formatting, and `pre-commit` to enforce standards.

- **Check and format:**

  ```sh
  ruff check . --fix
  ruff format .
  ```

- **Pre-commit:**
  ```sh
  pre-commit install
  pre-commit run --all-files
  ```

## Docker

A multi-stage `Dockerfile` is provided for building a minimal production image.

1.  **Build the image:**

    ```sh
    docker build -t skypilot-sidecar .
    ```

2.  **Run the container:**
    This command maps port `8000` on the host to `5555` in the container and mounts the default SkyPilot catalog directory.

    ```sh
    docker run -d -p 8000:5555 \
      -v ~/.sky/catalogs/v7:/root/.sky/catalogs/v7 \
      --name skypilot-sidecar-app \
      skypilot-sidecar
    ```

## Contributing

Contributions are welcome. Please follow the standard pull request workflow:

1.  Fork the repository.
2.  Create a new branch.
3.  Commit your changes.
4.  Push to your branch and open a pull request.
5.  Ensure all checks pass.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
