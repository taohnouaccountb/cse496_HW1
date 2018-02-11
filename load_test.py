import tensorflow as tf

path_prefix = '.\output\homework_1-0-107550'
print('test')

if __name__ == '__main__':
    with tf.Session() as session:
        graph = session.graph
        # loading the meta graph re-creates the graph structure in the current session, and restore initializes saved variables
        saver = tf.train.import_meta_graph(path_prefix + '.meta')
        saver.restore(session, path_prefix)

        # get handles to graph Tensors, noticing the use of name scope in retrieving model_output
        x = graph.get_tensor_by_name('input_placeholder:0')
        output = graph.get_tensor_by_name('output:0')
