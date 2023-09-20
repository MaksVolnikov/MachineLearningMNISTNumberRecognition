# MachineLearningMNISTNumberRecognition
Эта программа позволяет распознавать ЧБ изображенмя, размер которых составляет не более 3x3 с помощью модели персептрона, обученной на базе изображений MNIST.
В  приложении  реализована возможность  задания обучающей выборки из внешних файлов изображений.  
Изображения, используемые для распознавания, должны быть черно-белыми (bitmap) и размером не менее 9 пикселей.  
Программа имеет  два  режима  работы:  обучение  и распознавание.  
Обучение производиться по стандартному алгоритму обучения  персептрона с использованием дельта-правила.  
В программе задаются следующие настройки: 
– количество входов нейрона, которое соответствует общему числу  
пикселей изображения,  
– коэффициент скорости обучения (постоянное значение),  
– правильные варианты элементов обучающей выборки,  
– размер ошибки, при котором обучение персептрона завершается. 
На экранной форме приложения в режиме обучения персептрона отображаются:  
– элементы обучающей выборки (изображения),  
– настройки алгоритма обучения,  
– текущие (итоговые) веса нейронов и значение порога активационной 
функции,  
– протоколы  результатов  обучения (значения  весов  для  каждой 
итерации). 
На экранной форме режима распознавания персептрона отображаются:  
– распознаваемое  изображение  (можно выбрать из множества),  
– результат распознавания,  
– веса нейронов и значение порога активационной функции,  
– значения  выходов  всех  нейронов  до  и  после  применения 
активационной функции. 
