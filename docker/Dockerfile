ARG FROM_IMAGE=vnv_base
FROM ${FROM_IMAGE}:latest


# Label it 
ARG COMMIT_HASH=unknown
LABEL vnv.version.asgard=${COMMIT_HASH}


#Install asgard
COPY ./ ${SOURCE_DIR}/asgard

#Build it
RUN cd ${SOURCE_DIR}/asgard \
    && mkdir -p build \
    && cd build \
    && cmake -DInjection_DIR=${VNV_DIR}/lib/cmake -DASGARD_USE_VNV=ON .. \
    && make 
   

#Add the asgard config file to the vnv list of config files.
ENV VNV_CONFIG=${VNV_CONFIG}:${SOURCE_DIR}/asgard/build/vnv.__registration__

