import tensorflow as tf
import math


def conv3d_block(input, filters, kernel, symmetric, name, stride=1):
    with tf.variable_scope(name):
        if symmetric:
            conv = tf.layers.conv3d(input, filters, kernel_size= (kernel, kernel, kernel),strides=(stride, stride, stride), padding= "SAME")
            out = tf.layers.leaky_relu(conv)
            return out
        else:
            conv1 = tf.layers.conv3d(input, filters, kernel_size= (1, 1, kernel), strides=(stride, stride, stride), padding="SAME")
            act1 = tf.nn.relu(conv1)
            conv2 = tf.layers.conv3d(act1, filters, kernel_size= (1, kernel, 1), strides=(stride, stride, stride), padding="SAME")
            act2 = tf.nn.relu(conv2)
            conv = tf.layers.conv3d(act2, filters, kernel_size= (kernel, 1, 1), strides=(stride, stride, stride), padding="SAME")
            out = tf.nn.leaky_relu(conv)
            return out


def inception3d_block(input, filters, symmetric, block_name):
    with tf.variable_scope(block_name):
        if symmetric:
            #first path has a visual field of 3x3x3
            conv_reduce_1 = conv3d_block(input, filters= math.floor(filters/3), kernel= 1, symmetric= True, name= "dim_red1" )
            conv1 = conv3d_block(conv_reduce_1, filters= math.floor(filters/3), kernel= 3, symmetric= True, name= "conv1_3_3_3")
            # second path has a visual field of 5x5x5
            conv_reduce_2 = conv3d_block(input, filters=math.floor(filters / 3), kernel=1, symmetric=True)
            conv2_0 = conv3d_block(conv_reduce_2, filters=math.floor(filters / 3), kernel=3, symmetric=True)
            conv2_1 = conv3d_block(conv2_0, filters=math.floor(filters / 3), kernel=3, symmetric=True)
            # third path has a visual field of 1x1x1
            conv_reduce_3 = conv3d_block(input, filters=math.floor(filters / 3) + 1, kernel=1, symmetric=True)
            # Concatenation
            out = tf.concat([conv1, conv2_1, conv_reduce_3], axis=4, name="concat")
            return out
        else:
            # first path has a visual field of 3x3x3
            conv_reduce_1 = conv3d_block(input, filters=math.floor(filters / 3), kernel=1, symmetric=False)
            conv1 = conv3d_block(conv_reduce_1, filters=math.floor(filters / 3), kernel=3, symmetric=False)
            # second path has a visual field of 5x5x5
            conv_reduce_2 = conv3d_block(input, filters=math.floor(filters / 3), kernel=1, symmetric=False)
            conv2 = conv3d_block(conv_reduce_2, filters=math.floor(filters / 3), kernel=5, symmetric=False)
            # third path has a visual field of 1x1x1
            conv_reduce_3 = conv3d_block(input, filters=math.floor(filters / 3) + 1, kernel=1, symmetric=False)

            # Concatenation
            out = tf.concat([conv1, conv2, conv_reduce_3], axis=4, name="concat")
            return out


def normal3d_block(input, filters, symmetric, block_name, encode):
    with tf.variable_scope(block_name):
        if symmetric:
            if encode:
                filters1 = filters/2
            else:
                filters1 = filters*2
            conv1 = conv3d_block(input, filters= filters1, symmetric= True, kernel= 3)
            conv2 = conv3d_block(conv1, filters= filters, symmetric= True, kernel= 3)
            return conv2
        else:
            conv1 = conv3d_block(input, filters=filters, symmetric=False, kernel=5)
            return conv1


def conv3d_downsample(input, filters, block_name):
    with tf.variable_scope(block_name):
        down_conv = tf.layers.conv3d(input, filters= filters, kernel_size=(2,2,2), strides = (2,2,2), padding= "VALID")
        act = tf.nn.leaky_relu(down_conv)
        return act


def conv3d_upsample(input, filter, block_name):
    with tf.variable_scope(block_name):
        up_conv = tf.nn.conv3d_transpose(input, filter=[2, 2, 2, filter, filter], strides = [1, 2, 2, 2, 1], padding= "VALID")
        act = tf.nn.leaky_relu(up_conv)
        return act


def generator(input, filters=64, net_name= "Unet"):
    with tf.variable_scope(net_name):
        # Encoder
        encode1 = normal3d_block(input, filters, True, "Encode1", True)
        down1 = conv3d_downsample(encode1, filters, "Down1")
        encode2 = normal3d_block(down1, filters*2, True, "Encode2", True)
        down2 = conv3d_downsample(encode2, filters*2, "Down2")
        encode3 = normal3d_block(down2, filters*4, True, "Encode3", True)
        down3 =conv3d_downsample(encode3, filters*4, "Down3")

        # BottleNeck
        bottleneck = normal3d_block(down2, filters*4, True, "Bottleneck", True)

        #Decoder
        up1 = conv3d_upsample(bottleneck, filters*4, "Up1")
        res1 = tf.add(up1, encode3)
        decode1 = normal3d_block(res1, filters*2, True, "Decode1", False)
        up2 = conv3d_upsample(decode1, filters*2, "Up2")
        res2 = tf.add(up2, encode2)
        decode2 = normal3d_block(res2, filters*1, True, "Decode2", False)
        up3 = conv3d_upsample(decode2, filters*1, "Up3")
        res3 = tf.add(up3, encode1)
        decode3 = normal3d_block(res3, filters/2, True, "Decode3", False)
        out = tf.layers.conv3d(decode3, 3, kernel_size=(3,3,3), strides=(1,1,1), padding="same")
        return out
