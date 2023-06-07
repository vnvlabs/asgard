
ARG FROM_IMAGE=vnv_base
FROM ${FROM_IMAGE}


#Install asgard directly into the software dir as they dont have a install target.
COPY ./ ${SOFTWARE_DIR}/asgard

#Build it
RUN cd ${SOFTWARE_DIR}/asgard \
    && mkdir -p build \
    && cd build \
    && cmake -DInjection_DIR=${VNV_DIR}/lib/cmake -DASGARD_USE_VNV=ON .. \
    && make 
   
#Add the asgard config file to the vnv list of config files.
ENV VNV_CONFIG=${VNV_CONFIG}:${SOFTWARE_DIR}/asgard/vnv/config/vnv.__registration__ 


