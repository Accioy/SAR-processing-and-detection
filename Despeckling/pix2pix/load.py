from tensorflow.keras.models import load_model


model=load_model('combined0.h5')
model_a=model.get_layer('model_1')
model_b=model.get_layer('model')
model.summary()
model_a.summary() #generator
model_b.summary() #discriminator
model_b.save('discriminator0.h5')