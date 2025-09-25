variable "image_name" {
  description = "Docker image to pull from registry (e.g., username/iris-api:latest)"
  type        = string
}

variable "container_name" {
  description = "Name for the running container"
  type        = string
  default     = "iris-api"
}

variable "host_port" {
  description = "Host port to expose the FastAPI service"
  type        = number
  default     = 8000
}

variable "mlflow_tracking_uri" {
  description = "Tracking URI the container should use"
  type        = string
  default     = ""
}

variable "mlflow_experiment" {
  description = "MLflow experiment name to read from"
  type        = string
  default     = "iris_classification"
}

variable "model_uri" {
  description = "Optional explicit MLflow model URI"
  type        = string
  default     = ""
}

variable "labels_uri" {
  description = "Optional explicit MLflow artifact URI for label encoder classes"
  type        = string
  default     = ""
}

variable "run_id" {
  description = "Run ID metadata to surface in the API"
  type        = string
  default     = ""
}
