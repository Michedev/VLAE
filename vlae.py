def make_encoder():
    droput_rate = 0.05
    inputs = Input((28,28,1))
    with tf.name_scope('h_1'):
        h_1_layers = Sequential([ 
            Input((28, 28, 1)),
            Conv2D(8, 3),
            BatchNormalization(trainable=False),
            ReLU(),
            Conv2D(16, 3),
            BatchNormalization(trainable=False),
            SpatialDropout2D(droput_rate),
            ReLU()], name='h_1')
        h_1 = h_1_layers(inputs)
        h_1_flatten = Flatten()(h_1)
    with tf.name_scope('h_2'):
        h_2_layers = Sequential([ 
            Conv2D(16, 3),
            BatchNormalization(trainable=False),
            ReLU(),
            Conv2D(16, 3),
            BatchNormalization(trainable=False),
            SpatialDropout2D(droput_rate),
            ReLU()], name='h_2')
        h_2 = h_2_layers(h_1)
        h_2_flatten = Flatten()(h_2)
    with tf.name_scope('h_3'):
        h_3_layers = Sequential([ 
            Conv2D(16, 3),
            BatchNormalization(trainable=False),
            ReLU(),
            Conv2D(16, 3),
            BatchNormalization(trainable=False),
            SpatialDropout2D(droput_rate),
            ReLU()], name='h_3')
        h_3 = h_3_layers(h_2)
        h_3_flatten = Flatten()(h_3)
    return Model(inputs, [h_1_flatten, h_2_flatten, h_3_flatten], name='encoder')
        
def make_decoder(latent_dim1, latent_dim2, latent_dim3):
    z_1_input, z_2_input, z_3_input = Input((latent_dim1,), name='z_1'), Input((latent_dim2,), name='z_2'), Input((latent_dim3,), name='z_3')
    
    with tf.name_scope('z_tilde_3'):
        z_3 = Dense(1024, activation='relu')(z_3_input)
        z_tilde_3_layers = Sequential([
            Dense(1024),
            BatchNormalization(trainable=False),
            ReLU()] * 3, name='z_tilde_3')
        z_tilde_3 = z_tilde_3_layers(z_3)
        
    with tf.name_scope('z_tilde_2'):
        z_2 = Dense(128, activation='relu')(z_2_input)
        z_tilde_2_layers = Sequential([
            Dense(128),
            BatchNormalization(trainable=False),
             ReLU()] * 3, name='z_tilde_2')
        input_z_tilde_2 = Concatenate()([z_tilde_3, z_2])
        z_tilde_2 =  z_tilde_2_layers(input_z_tilde_2)
    
    with tf.name_scope('z_tilde_1'):
        z_1 = Dense(128, activation='relu')(z_1_input)
        z_tilde_1_layers = Sequential([
            Dense(128),
            BatchNormalization(trainable=False),
             ReLU()] * 3, name='z_tilde_1')
        input_z_tilde_1 = Concatenate()([z_tilde_2, z_1])
        z_tilde_1 =  z_tilde_1_layers(input_z_tilde_1)
        
    with tf.name_scope('decoder'):
        decoder = Reshape((2,2,32))(z_tilde_1)
        decoder = UpSampling2D(2)(decoder) #4x4
        decoder = Conv2D(32, 3)(decoder) #2x2
        decoder = BatchNormalization(trainable=False)(decoder)
        decoder = Activation(tf.nn.crelu)(decoder)
        decoder = UpSampling2D(4)(decoder) #8x8
        decoder = Conv2D(16, 3)(decoder) #6x6
        decoder = BatchNormalization(trainable=False)(decoder)
        decoder = Activation(tf.nn.crelu)(decoder)
        decoder = UpSampling2D(2)(decoder) #12x12
        decoder = Conv2D(8, 3)(decoder) #10x10
        decoder = BatchNormalization(trainable=False)(decoder)
        decoder = Activation(tf.nn.crelu)(decoder)
        decoder = UpSampling2D(2)(decoder) #20x20
        decoder = Conv2D(4, 5)(decoder) #16x16
        decoder = BatchNormalization(trainable=False)(decoder)
        decoder = LeakyReLU()(decoder)
        decoder = UpSampling2D(2)(decoder) #32x32
        decoder = Conv2D(1, 5)(decoder) #28x28
        decoder = Activation('sigmoid')(decoder)
    return Model([z_1_input, z_2_input, z_3_input], decoder, name='decoder')

def make_vlae(latent_size):
    with tf.name_scope('encoder'):
        encoder = make_encoder()
    with tf.name_scope('decoder'):
        decoder = make_decoder(latent_size, latent_size, latent_size)
    inputs = Input((28,28,1))
    h_1, h_2, h_3 = encoder(inputs)
    z_1 = NormalVariational(latent_size, add_kl=False, coef_kl=0.0, add_mmd=True, lambda_mmd=1., name='z_1_latent')(h_1)
    z_2 = NormalVariational(latent_size, add_kl=False, coef_kl=0.0, add_mmd=True, lambda_mmd=1., name='z_2_latent')(h_2)
    z_3 = NormalVariational(latent_size, add_kl=False, coef_kl=0.0, add_mmd=True, lambda_mmd=10., name='z_3_latent')(h_3)
    
    decoded = decoder([z_1, z_2, z_3])
    vlae = Model(inputs, decoded, name='vlae')
    return vlae