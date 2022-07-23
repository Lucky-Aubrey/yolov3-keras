import tensorflow as tf
from keras import Input
from keras.layers import (
    UpSampling2D, Concatenate, Lambda
    )
from model import (
    conv_block, res_block, conv_block_2
    )

def darknet(name):
    
    x = inputs = Input([None, None, 3])
    x = conv_block(inputs, 32, 3)
    x = conv_block(x, 64, 3, 2)
    x = res_block(x, [32, 64], 1)
    
    x = conv_block(x, 128, 3, 2)
    x = res_block(x, [64, 128], 2)
    
    x = conv_block(x, 256, 3, 2)
    x = x_36 = res_block(x, [128, 256], 8)
    
    x = conv_block(x, 512, 3, 2)
    x = x_61 = res_block(x, [256, 512], 8)
    
    x = conv_block(x, 1024, 3, 2)
    x = res_block(x, [512, 1024], 4)
    return tf.keras.Model(inputs, [x_36, x_61, x], name=name)

# TODO
def yolo_conv(x_in, filters, kernels, strides, filters_up, name):
    if isinstance(x_in, tuple):
        inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
        x, x_skip = inputs

        # concat with skip connection
        x = conv_block(x, filters_up, 1, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, x_skip])
    else:
        x = inputs = Input(x_in.shape[1:])
    
    x = conv_block_2(x, filters, kernels, strides)
    return tf.keras.Model(inputs, x, name=name)(x_in)

def yolo_output(x_in, filters, anchors, num_classes, name):
    x = inputs = Input(x_in.shape[1:])
    x = conv_block(x, filters, 3, 1)
    x = conv_block(x, (5+num_classes)*3, 1, 1, False)
    x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, num_classes + 5)))(x)
    return tf.keras.Model(inputs, x, name=name)(x_in)


def build_model(num_classes, shape=(416, 416, 3)):
    
    x = inputs = Input(shape=(None, None, 3), name='input')
    
    #Darknet53
    x_36, x_61, x = darknet('yolo_darknet')(x)
    
    # 13 by 13 detection head
    x = yolo_conv(x, [512, 1024, 512, 1024, 512, ], 
                      [1,3,1,3,1], 
                      [1,1,1,1,1],
                      None,
                      name='yolo_conv_0')
    
    out0 = yolo_output(x, 1024, 3, num_classes, name='yolo_output_0')
    
    # 26 by 26 detection head
    
    x = yolo_conv((x, x_61), [256, 512, 256, 512, 256, ], 
                      [1,3,1,3,1], 
                      [1,1,1,1,1],
                      256,
                      name='yolo_conv_1')
    
    out1 = yolo_output(x, 512, 3, num_classes, name='yolo_output_1')

    
    # 52 by 52 detection head
    x = yolo_conv((x,x_36), [128, 256, 128, 256, 128, ], 
                      [1,3,1,3,1], 
                      [1,1,1,1,1],
                      128,
                      name='yolo_conv_2')
    
    out2 = yolo_output(x, 256, 3, num_classes, name='yolo_output_2')
    
    outputs = [out0, out1, out2]
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='darknet_53')
