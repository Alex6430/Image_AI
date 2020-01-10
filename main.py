# from imageai.Detection import ObjectDetection
# import os
#
# exec_path = os.getcwd()
#
# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath(os.path.join(exec_path, "resnet50_coco_best_v2.0.1.h5"))
# detector.loadModel()
#
# list = detector.detectObjectsFromImage(input_image=os.path.join(exec_path, "object.jpg")
#                                        , output_image_path=os.path.join(exec_path, "new_object.jpg")
#                                        )

from Recognition import *

if __name__ == '__main__':
    image_file = "C:\\Users\\hp\\Desktop\\samples\\"+"numbers1.jpg"
    # letters_extract_old(image_file)


    # cv2.imshow("0", letters_extract(image_file)[0][2])
    # cv2.imshow("1", letters_extract(image_file)[1][2])
    # cv2.imshow("2", letters_extract(image_file)[2][2])
    # cv2.imshow("3", letters_extract(image_file)[3][2])
    # cv2.imshow("4", letters_extract(image_file)[4][2])
    # cv2.imshow("5", letters_extract(image_file)[5][2])
    # cv2.waitKey(0)

    # model = emnist_model()
    #
    # # Set a learning rate reduction
    # learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5,
    #                                                             min_lr=0.00001)
    #
    # # Required for learning_rate_reduction:
    # keras.backend.get_session().run(tf.global_variables_initializer())
    #
    # model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction],
    #           batch_size=64, epochs=30)
    #
    # model.save('emnist_letters.h5')


    model = keras.models.load_model('emnist_letters.h5')
    s_out = img_to_str(model, image_file)
    print(s_out)

