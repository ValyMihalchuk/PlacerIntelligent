# intelligent-placer
Требуется написать Intelligent Placer, который по поданной фотографии с несколькими предметами на светлой горизонтальной поверхности с многоугольником на белом листе А4 определяет, можно ли поместить эти предметы внутрь многоугольника.
Более точно, Intelligent Placer должен решить, существует ли способ разместить эти предметы в многоугольнике так, чтобы ни один из предметов не выходил за границы многоугольника и к тому же, ни один из этих предметов не накрывал бы собой другие. В ходе описанного размещения предметов мы не должны использовать области внутри них самих, то есть предметы рассматриваюся как единое целое, без дырок.
# Постановка задачи
### Общие
- На вход Intelligent Placer поступает путь до фотографии (с предметами и прямоугольником)
- Программа отвечает True если предметы помещаются в многоугольник, False иначе.
- Ответ выводится в стандартный поток вывода
### Многоугольник
- Многоугольник должен быть замкнутым и выпуклым. В данном случае, под многоугольником подразумевается лишь замкнутая ломаная, а не вся часть плоскости, ею ограниченная.
- Границы многоугольника должны быть четко видны и нарисованы темным маркером на белом листе бумаги. Толщина линий ломанной не должна быть меньше *1 мм*

### Содержание фотографий
- Предметы и лист с многоугольником должны быть размещены на светлой плоской горизонтальной поверхности, с однообразной текстурой, не сливающейся с листом.
- Предметы должны располагаться вне многоугольника, так, что они не должны пересекаться ни с друг с другом, ни с многоугольником
- Объекты могут быть только из заранее определенного множества (определенного в папке pictures)
- Один предмет не может присутствовать на фото несколько раз

### Фотометрические
- фотографии должны быть в формате *\*.jpg*
- угол между направлением камеры и перпендикуляром к поверхности должен быть не более *10&deg;*
- расстояние между камерой и поверхностью должно быть таким, чтобы на получившейся фотографии были легко отличимы предметы и видна была граница. Для этого рекомендуется поддерживать данное расстояние не более *0.5 м*
- фотография должны быть снята камерой с штатным объективом либо объективом с фокусным расстоянием, слегка превышающим нормальное. Иными словами, на полученном изображении должна быть обеспечена наиболее естественная перспектива
- на изображении не должно быть длинных теней
- фотография должна быть сделана при освещенности, не противоречащей нормам СанПиН 2.2.1/2.1.1.1278-03. Другими словами, освещенность должна максимально естественная для человеческого глаза.
- фотография резкая, не смазанная


# План
0. Подготавливаем изображение: переводим его в ч/б.
1. Находим границы листа 
    * используем морфологические операции и бинаризацию Оцу
    * лист  — максимальная по площади компонента связности[^0]
    * другая идея - найти границы листа с помощью преобразования Хафа (мы найдем прямые, ограничивающие лист)
2. Зная границы листа (bbox из regionprops[^1]) находим уже внутри него многоугольник с помощью бинаризации.
    * далее можно применить метод skimage.measure.find_contours (на маске либо сразу), чтобы найти вершины этого многоугольника
3. Находим предметы
    * вычитаем фон
    * применяем фильтр гаусса 
    * используем бинаризацию (а так же локальную)[^2]
    * находим компоненты связности предметов с помощью сортировки по площадям компонент связности[^3]
4. Для каждой такой компоненты связности находим ее ограничивающую рамку (bbox из regionprops[^1])
5. Далее задача сводится к задаче упаковки прямоугольников в многоугольник. 
    * Один из вариантов - использовать **жадный алгоритм** - возьмем прямоугольник с самой большой площадью и поместим его внутрь многоугольника (логичнее поместить его к границам). Проверяем, поместился ли он внутрь (то есть, не вышел ли он за границы - а его границы мы уже знаем). Если поместился, повторяем эти действия вновь - берем следующий по площади прямоугольник и помещаем его внутрь многоугольника. Если в какой-то момент прямоугольник вышел за границы многоугольника - возвращаем False. Если все прямоугольники поместились - возвращаем True.
       * Вместо ограничивающего прямоугольника можно использовать многоугольник (skimage.measure.find_contours)
       * Для каждой попытки разместить прямоугольник/многоугольник, в случае если он не помещается, можно пробовать его поворачивать на углы с заданным шагом
       * Вместо того, чтобы находить границы, возможно целесообразнее использовать маски (и фигуры и предметов). Алгоритм будет таким же.


[^0]: в некоторых случаях, бинаризация может сработать плохо: вместо одного только листа по максимальной площади может определиться лист+шумы или что-то совсем другое. Возможно, стоит включить сюда другие способы определения листа  — по его расположению (например, лист всегда слева - тогда такое требование нужно будет включить в постановку задачи). Либо, поэкспеременировать над преобработкой для этого этапа - попробовать подбирать параметры фильтра Гаусса.
[^1]: из пакета skimage (skimage.measure)
[^2]: возможно, здесь логичнее будет использовать Canny
[^3]: если разрешить алгоритму использовать изображения предметов, то мы будем знать их минимальный и максимальный размер относительно листа А4.

# Данные
- [Изображения объектов](pictures)
- [Тесты](tests)
- [Разметка тестов](tests/tests.csv)
- [Описание тестов](tests/description.txt)
