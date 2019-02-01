//output = x + 2y;

let tf = require('@tensorflow/tfjs-node');

const model = tf.sequential();

//configure

// Layer 1
const hidden = tf.layers.dense({
    units: 5,
    activation:"elu",
    inputShape:[2] // Input layer [2] 1d array with 2 indexes
});
//Layer 2
const output = tf.layers.dense({
    units: 1,
    activation: "elu"
});

model.add(hidden);
model.add(output);

//compile
const sgdConfig = tf.train.sgd(0.1); //starchastic gradient descent
model.compile({
    loss: tf.losses.huberLoss,
    optimizer: sgdConfig
});

//fit/train

const xs = tf.tensor2d([
    [3,4],
    [1,3],
    [1, 2],
    [4, 8],
    [2, 5],
    [7, 7],
]);


const ys = tf.tensor2d([
    [11],
    [7],
    [5],
    [20],
    [12],
    [21],
]);

train().then(()=> {
    console.log("training complete...")
    //predict
    const xp = tf.tensor2d([
        [0,1],
        [3, 3],
        [2, 4],
    ]);
    let predictions = model.predict(xp);
    predictions.print();
});

async function train(params) {
    for(var i=0; i< 1000; i++) {
        await model.fit(xs, ys, {
            shuffle: true,
            epochs: 5
        });
    }
}



