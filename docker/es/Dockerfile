ARG ES_IMAGE=docker.io/library/elasticsearch:8.11.3
FROM ${ES_IMAGE}

ARG IK_VERSION=8.11.3

RUN bin/elasticsearch-plugin install --batch "https://get.infini.cloud/elasticsearch/analysis-ik/${IK_VERSION}"
