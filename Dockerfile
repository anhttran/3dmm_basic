FROM mxnet/python:1.1.0
WORKDIR /app
ADD . /app

RUN apt-get update
RUN apt-get install -y libhdf5-serial-dev libboost-all-dev nano cmake libosmesa6-dev freeglut3-dev awscli zip
RUN pip3 uninstall -y requests
RUN pip3 install requests==2.12

RUN mkdir build; \
	cd build; \
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../bin ..; \
	make; \
	make install; \
	cd ..

RUN cp /app/lib/BaselFace.dat /app/bin/

WORKDIR /app/bin
EXPOSE 80

ENV NAME World

CMD ["bash", "IRIS_3DMM"]
