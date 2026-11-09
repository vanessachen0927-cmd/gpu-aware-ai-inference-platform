variable "kubeconfig" {
  type    = string
  default = "~/.kube/config"
}

variable "namespace" {
  type    = string
  default = "ai-inference"
}

variable "image_repository" {
  type    = string
  default = "gpu-inference"
}

variable "image_tag" {
  type    = string
  default = "0.1.0"
}

provider "kubernetes" {
  config_path = pathexpand(var.kubeconfig)
}

provider "helm" {
  kubernetes {
    config_path = pathexpand(var.kubeconfig)
  }
}

resource "kubernetes_namespace" "inference" {
  metadata {
    name = var.namespace
  }
}

resource "helm_release" "inference" {
  name      = "gpu-inference"
  namespace = kubernetes_namespace.inference.metadata[0].name
  chart     = "../helm/gpu-inference"

  set {
    name  = "image.repository"
    value = var.image_repository
  }

  set {
    name  = "image.tag"
    value = var.image_tag
  }
}
