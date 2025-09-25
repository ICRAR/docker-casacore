
# DOCKER-CASACORE

This small project packs the various docker builds required to get a working casacore
installation with the Adios2StMan and MGARD compression enabled. Dysco is also enabled
and thus this image is well suited to investigate the compression of visibilities.

See also 
- [casacore](https://github.com/casacore/casacore) The casacore GIT repo.
- [ADIOS2](https://github.com/ornladios/ADIOS2) The ADIOS2 advanced I/O system.
- [MGARD](https://github.com/CODARcode/MGARD) The MGARD compression suite.

## Usage
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
make docker-start [MY_CMD=<cmd>]
```
This make target allows to override the default launching of ipython inside the docker container
using the MY_CMD variable. For example
```bash
make docker-start MY_CMD=bash
```
will start a bash shell in the container. In the same way it is also possible to place an *executable* python script into the ~/scratch directory and execute it
```bash
make docker-start MY_CMD=/scratch/my_script.py
```
There is also a stop target available, which cleans up potentially 
orphan containers, although this should not happen.
```bash
make docker-stop
```

