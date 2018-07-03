console.clear();

// activation ('elu'|'hardSigmoid'|'linear'|'relu'|'relu6'| 'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh'|string) Name of the activation function to use.

const model = tf.sequential({layers:[                                     // adam: elu , selu, sigmoid, tanh 
    tf.layers.dense({units: 4, inputShape: 2, activation: 'tanh'}),       // sgd: 'elu', 'selu', tanh
    tf.layers.dense({units: 1, activation: 'tanh'})                       // rmsprop: 'relu'
]});
model.compile({ optimizer: tf.train.adam(0.1), loss: 'meanSquaredError' });

const xs = tf.tensor2d([[0,0], [0,1], [1,0], [1,1]]);
const ys = tf.tensor2d([[0],[1],[1],[0]]);
model.fit(xs, ys, {epochs: 100, shuffle: true}).then(() => {
   model.predict(xs).print();
});
