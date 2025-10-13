.ONESHELL:
MY_GID=$(shell id -g)
MY_UID=$(shell id -u) 
MY_VOLUME=$(HOME)/scratch
TARGET=ubuntu-base
CONTAINER=casacore
CMD=ipython

.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo ""
	@fgrep "##" Makefile | fgrep -v fgrep | sed "s/##//g"

.PHONY : build-all build-ubuntu-base build-adios2-mgard build-casacore build-wsclean
##build-all:         Build the complete stack of images. This takes very long.
build-all : build-ubuntu-base build-adios-mgard build-casacore build-wsclean

.PHONY : build
build:            ## Build an image 
	@if ! command -v docker; then echo "Docker is not available; please confirm it is installed." && exit; fi
	@docker build -f docker/build_$(TARGET).docker --tag icrar/$(TARGET) .

build-ubuntu-base:## Build the ubuntu base image
	$(MAKE) build TARGET="ubuntu-base"

build-adios-mgard:## Build the adios2-mgard image
	$(MAKE) build TARGET="adios2-mgard"

build-casacore:   ## Build the casacore image
	$(MAKE) build TARGET="casacore"

build-wsclean:    ## Build the wsclean image
	$(MAKE) build TARGET="wsclean"

.PHONY: start
start:            ## run an image image with optional CMD variable
	@mkdir -p $(HOME)/scratch
	@if ! command -v docker; then echo "Docker is not available; please confirm it is installed." && exit; fi
	@MY_GID=$(MY_GID) MY_UID=$(MY_UID) CMD=$(CMD) MY_VOLUME=$(MY_VOLUME) docker compose -f docker/docker-compose-$(CONTAINER).yaml run --rm $(CONTAINER)

start-casacore:   ## start the wsclean image into ipython
	$(MAKE) start CONTAINER=casacore CMD=ipython

start-wsclean:   ## start the wsclean image into bash
	$(MAKE) start CONTAINER=wsclean CMD=bash

.PHONY: stop
stop:             ## stop the casacore container
	@MY_GID=$(MY_GID) MY_UID=$(MY_UID) CMD=$(CMD) MY_VOLUME=$(MY_VOLUME) docker compose -f docker/docker-compose.yaml run --rm --remove-orphans casacore echo

.PHONY: stop-wsclean
stop-wsclean:     ## stop the wsclean container
	@MY_GID=$(MY_GID) MY_UID=$(MY_UID) CMD=$(CMD) MY_VOLUME=$(MY_VOLUME) docker compose -f docker/docker-compose-wsclean.yaml run --rm --remove-orphans wsclean echo

.PHONY: test
test:             ## Run a simple test
	@mkdir -p $(HOME)/scratch
	@echo "Creating a small MS using the Adios2StMan"
	@MY_GID=$(MY_GID) MY_UID=$(MY_UID) MY_VOLUME=$(MY_VOLUME) CMD=$(CMD) docker compose -f docker/docker-compose.yaml run --rm casacore /code/casacore/build/tables/DataMan/test/tAdios2StMan > /dev/null
	@echo "Checking with python-casacore"
	@MY_GID=$(MY_GID) MY_UID=$(MY_UID) MY_VOLUME=$(MY_VOLUME) CMD=/code/test_Adios2StMan.py docker compose -f docker/docker-compose.yaml run --rm casacore

.PHONY: release
release:          ## Create a new tag for release.
	@echo "WARNING: This operation will create s version tag and push to github"
	@read -p "Version? (provide the next x.y.z semver) : " TAG
	@if ! grep -q "v$${TAG}" CHANGELOG.md; then echo "TAG version number must be added to CHANGELOG.md before committing." && exit; fi
	@echo "v$${TAG}" > VERSION
	@git add VERSION CHANGELOG.md
	@git commit -m "Release: version v$${TAG} 🚀"
	@echo "creating git tag : v$${TAG}"
	@git tag v$${TAG}
	@git push -u origin HEAD --tags
#	@echo "Github Actions will detect the new tag and release the new version."

# This Makefile has been based on the existing ICRAR/daliuge-component-template.
# __author__ = 'ICRAR'
# __repo__ = https://github.com/ICRAR/daliuge
# __sponsor__ = https://github.com/sponsors/ICRAR/
