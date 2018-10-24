

from keras.models import Model
from keras.layers import *
from keras.optimizers import SGD

import mol
import trainer

import os


class MolPredictModel(object):

    def __init__(self):

        self.grid_span = 20
        
        self.atom_types = len(trainer.ATOM_TYPES)

        self.target_grid_shape = (self.grid_span, self.grid_span, self.grid_span, self.atom_types)
        self.ligand_grid_shape = (self.grid_span, self.grid_span, self.grid_span, self.atom_types)
        self.fragment_grid_shape = (self.grid_span, self.grid_span, self.grid_span, 1)

        self.model = self.build()

    def build(self):
        '''
        Build the model
        '''
        self.target_in = Input(shape=self.target_grid_shape, name='target_in')
        self.ligand_in = Input(shape=self.ligand_grid_shape, name='ligand_in')

        # concatenate
        x = Concatenate(axis=4)([self.ligand_in, self.target_in])

        # downsampling layers
        x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
        x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
        x = MaxPool3D((2,2,2))(x)
        x = Conv3D(128, (3,3,3), padding='same', activation='relu')(x)
        x = Conv3D(128, (3,3,3), padding='same', activation='relu')(x)
        x = MaxPool3D((2,2,2))(x)
        x = Conv3D(256, (3,3,3), padding='same', activation='relu')(x)
        x = Conv3D(256, (3,3,3), padding='same', activation='relu')(x)
        
        # upsample
        # x = Deconv3D(1, (3,3,3), strides=2, padding='same', activation='relu')(x)
        x = Deconv3D(1, (5,5,5), strides=4, padding='same', activation='sigmoid', name='fragment_out')(x)

        self.fragment_out = x

        model = Model(inputs=[self.ligand_in, self.target_in], outputs=[self.fragment_out])

        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary(line_length=110)

        return model

    def train(self):

        targets = [x for x in os.listdir('../data/dude/') if x[0] != '.']

        train = targets[2:3]
        test = targets[-2:-1]

        print('Training targets:')
        for t in train:
            print('- %s' % t)

        print('Test targets:')
        for t in test:
            print('- %s' % t)
        
        print('Loading training data...')
        training_data = trainer.load_data(train)

        print('Loading testing data...')
        testing_data = trainer.load_data(test)

        print('Generating testing grid data...')
        full_testing_data = trainer.full_grid_generator(testing_data, 20, 1)

        print('Training...')
        hist = self.model.fit_generator(
            trainer.grid_generator(training_data, 5, 20, 1),
            steps_per_epoch=30,
            epochs=1,
            validation_data=full_testing_data
        )

