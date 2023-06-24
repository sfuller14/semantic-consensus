AWS_REGION ?= us-east-2
ECR_ENDPOINT ?= "551238779288.dkr.ecr.us-east-2.amazonaws.com"
IMAGE_NAME ?= "ecommerce-semantic-recsys"
TS=$(shell date +'%Y%m%d%H%M%S')
TAG=${IMAGE_NAME}:${TS}

deploy:## Rebuild docker image with a new tag and push to ECR
		docker buildx build --platform linux/amd64 --build-arg AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
             --build-arg AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
             --build-arg PINECONE_API_KEY=$(PINECONE_API_KEY) \
             --build-arg PINECONE_ENVIRONMENT=$(PINECONE_ENVIRONMENT) \
             --build-arg OPENAI_API_KEY=$(OPENAI_API_KEY) \
             --build-arg AWS_RDS_HOSTNAME=$(AWS_RDS_HOSTNAME) \
             --build-arg AWS_RDS_PORT=$(AWS_RDS_PORT) \
             --build-arg AWS_RDS_DB=$(AWS_RDS_DB) \
             --build-arg AWS_RDS_UN=$(AWS_RDS_UN) \
             --build-arg AWS_RDS_PW=$(AWS_RDS_PW) \
             --build-arg COHERE_KEY=$(COHERE_KEY) \
             --build-arg AWS_DEFAULT_REGION=$(AWS_DEFAULT_REGION) \
			 -t ${TAG} .
	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_ENDPOINT}
	docker tag ${TAG} ${ECR_ENDPOINT}/${TAG}
	docker push ${ECR_ENDPOINT}/${TAG}
	sed -i '' 's#^\( *image: *\).*#\1'${ECR_ENDPOINT}'/'${TAG}'#' docker-compose.yaml
	git add docker-compose.yaml
	git commit -m "autocommit: image version updated"
	eb deploy

.PHONY: rebuild
