ARG FROM_IMAGE=vnv_base
FROM ${FROM_IMAGE}:latest

#Install asgard
COPY ./ ./asgard
RUN apt-get install -y libomp-dev
RUN cd ./asgard && mkdir -p build && cd build && cmake -DInjection_DIR=${VNV_DIR}/lib/cmake -DASGARD_USE_VNV=ON .. && make 
#Run asgard
RUN cd ./asgard/build && ./asgard 

ENV VNV_CONFIG=${VNV_CONFIG}:/asgard/.vnv-config.json

ARG COMMIT_HASH=unknown
LABEL vnv.version.asgard=${COMMIT_HASH}
