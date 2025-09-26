
# DOCKER-CASACORE

This small project packs the various docker builds required to get a working casacore
installation with the Adios2StMan and MGARD compression enabled. Dysco is also enabled
and thus this image is well suited to investigate the compression of visibilities.

See also 
- [casacore](https://github.com/casacore/casacore) The casacore GIT repo.
- [ADIOS2](https://github.com/ornladios/ADIOS2) The ADIOS2 advanced I/O system.
- [MGARD](https://github.com/CODARcode/MGARD) The MGARD compression suite.

## Usage
The package comes with a Makefile allowing an easy and straight forward way to build and run
the image.

```bash
Usage: make <target>

Targets:

help:              Show the help.
build-all:         Build the complete stack of images. This takes very long.
build-ubuntu-base: Build the ubuntu base image
build-adios-mgard: Build the adios2-mgard image
build-casacore:    Build the casacore image
start:             run the final image with optional CMD variable
stop:              Install using docker containers
test:              Run a simple test
release:           Create a new tag for release.
```
### Complete build
This will take pretty long, but hopefuly the first two images don't have to be done too often. 

```bash
make build-all
```

It is also possible to build the individual images. Note that these are dependent on each other.

```bash
make build-ubuntu-base
make build-adios2-mgard
make build-casacore
```
The first image will be tagged `icrar/ubuntu-clang`, the second `icrar/adios2-mgard` and the final
one `icrar/casacore`.

### Starting and stopping the casacore image
The package also contains a docker-compose file to enable running the casacore image using the
current user. It will mount the directory ~/scratch into the container in order to be able to
exchange data with the host file system.

```bash
make start [CMD=<cmd>] [MY_VOLUME=<path>]
```
This make target allows to override the default launching of ipython inside the docker container
using the `CMD` variable. For example
```bash
make start CMD=bash
```
will start a bash shell in the container. In the same way it is also possible to place an *executable* python script into the ~/scratch directory and execute it
```bash
make start CMD=/scratch/my_script.py
```
Using the `MY_VOLUME` variable it is possible to mount any host directory as /scratch inside the container to allow the exchange of data between the host and the container. For example
```bash
make start MY_VOLUME=~/data
```
would mount the data directory under your home directory as /scratch inside the container.

There is also a stop target available, which cleans up potentially 
orphan containers, although this should not happen in normal circumstances.
```bash
make stop
```

There is also a test target, which produces two MeasurmentSets using the Adios2 storage manager from the casacore C++ level and reads the data back using python-casacore. This makes sure that most of the stack is working correctly.
```bash
make test
```
