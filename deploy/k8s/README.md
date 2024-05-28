# K8s deployment

A helm chart for the deployment of the project is provided in the root's `deploy/k8s/alive-models` directory.  
You can override any of the values in the [values.yaml](./models/values.yaml) file with the `--set` flag.  
A generated example manifest yaml is also available in the `deploy/k8s/manifest.example.yaml` file: [./manifest.example.yaml](./manifest.example.yaml).  
The values are also available in the `.env.example` file in the root of the project.  
They (a copy of them in `.env`) can be used if using the short commands provided in the [Makefile](../../Makefile).  
i.e. `make k8s-template` and (after inspecting the generated yaml file) `make k8s-apply`.

## Generate the manifest using helm

```shell
# on the root of the project (where the relative path `deploy/k8s/alive-models` exists):
helm template \
    --name-template=alive-models \
    --set namespace=alive \
    deploy/k8s/alive-models > deploy/k8s/manifest.yaml
# inspect the generated manifest, modify it if necessary
# To apply (--set) all the variables in the .env file:
# make k8s-template
```

### Apply the generated manifest

```shell
# make sure the namespace to use exists
NAMESPACE=alive
# if using minikube, you can:
# alias kubectl="minikube kubectl --"
kubectl create ns ${NAMESPACE}
# apply
kubectl apply -f manifest.yaml
```

## Minikube Example

To deploy the project in a minikube cluster, you can follow the instructions in the [minikube](https://minikube.sigs.k8s.io/docs/) documentation.  
For the nvidia device plugin, you can follow the instructions in the [nvidia](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/) tutorial.  
An example using the `none` driver is provided below:

```shell
minikube start \
    --driver=none \
    --apiserver-ips 127.0.0.1 \
    --apiserver-name localhost \
    --container-runtime=containerd \
    --insecure-registry=localhost
```

### Firewall

If a firewall is enabled, you may need to allow the ports used by the cluster. Example for `ufw`:

```shell
sudo ufw allow 2379/tcp
# find the device that is used by the cluster
NODE_IP="$(kubectl get no -o json | jq '.items[0].spec.podCIDR' | cut -d/ -f1 | tr -d '\"')"
echo "Node IP: ${NODE_IP}"
# Node IP: 10.244.0.0
IP_PART_TO_GREP="$(echo $NODE_IP | cut -d . -f1-2)"
echo "IP part to grep: ${IP_PART_TO_GREP}"
# IP part to grep: 10.244
DEVICE="$(ip -br a | grep UP | grep "${IP_PART_TO_GREP}" | head -1 | awk '{print $1}')"
echo "Device: ${DEVICE}"
# Device: cni0
# allow traffic on the device
sudo ufw allow in on "${DEVICE}" && sudo ufw allow out on "${DEVICE}"
```

### NVIDIA Device plugin

```shell
minikube addons enable nvidia-device-plugin
```

### Container registry

To push a local image to the minikube registry, run the following commands:

```shell
minikube addons enable registry
# start a temporary forwarder to the minikube registry
# docker/podman
CONTAINER_COMMAND=podman
${CONTAINER_COMMAND} run --rm -it --network=host \
    alpine ash -c "apk add socat && socat TCP-LISTEN:5000,reuseaddr,fork TCP:$(minikube ip):5000"
## on a new terminal/tab:
CONTAINER_COMMAND=podman
## tag if necessary to point to localhost:5000
## for example:
# ${CONTAINER_COMMAND} tag alive_models:latest-cuda-12.4.1 localhost:5000/alive_models:latest-cuda-12.4.1
## push the image
${CONTAINER_COMMAND} push localhost:5000/alive_models:latest-cuda-12.4.1
```
