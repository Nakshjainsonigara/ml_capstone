terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
  required_version = ">= 1.4.0"
}

provider "docker" {}

resource "docker_registry_image" "api" {
  name = var.image_name
}

resource "docker_image" "api" {
  name         = docker_registry_image.api.name
  keep_locally = true
}

resource "docker_container" "api" {
  name  = var.container_name
  image = docker_image.api.image_id
  restart = "unless-stopped"

  ports {
    internal = 8000
    external = var.host_port
  }

  env = [
    "MLFLOW_TRACKING_URI=${var.mlflow_tracking_uri}",
    "MLFLOW_EXPERIMENT_NAME=${var.mlflow_experiment}",
    "MODEL_URI=${var.model_uri}",
    "LABELS_URI=${var.labels_uri}",
    "RUN_ID=${var.run_id}",
  ]
}
