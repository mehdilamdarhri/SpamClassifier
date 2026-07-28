from app.predictor import predict_message

# Talks to the user
message = input("Enter a message: ")
result = predict_message(message)
print(result)