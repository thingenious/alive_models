# Podman / Docker Compose deployment

A compose file example for the deployment of the project is provided in the root's `deploy/compose` directory: [./compose.example.yaml](./compose.example.yaml).  
Make a copy of this file, named `compose.yaml`, (.gitgnored), and adjust the configuration to your needs.  
Do specify the gpu device based on the engine that you are using (examples are provided as comments in the example file).  
Make sure you have installed the NVIDIA Container Toolkit if you are using the NVIDIA runtime:  
For docker: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker>  
For podman: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html>

To deploy the project, run the following command in the root directory of the project:  

```bash
CONTAINER_COMMAND=podman # or docker
${CONTAINER_COMMAND} compose -f deploy/compose/compose.yaml up -d
## or you can use the action in Makefile:
# make compose-up
```

Note that on the first run, the containers will take some time to download the required model files.  
If a volume is used (default), the files will be stored in the volume and will not be downloaded again on subsequent runs.
