# Lambda

Deploy a Lambda function that responds with the model response.

- **Pros:** pay per request.
- **Cons:** slow cold start, quantized models.

## Lambda managed instances

Deploy to a GPU Lambda function.

- **Pros:** pay per request (you pay for instances).
- **Cons:** cold start when scaled to zero (where supported).
