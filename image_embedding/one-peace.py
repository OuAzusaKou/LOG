import dashscope

def image_call(image_path):
    # 构建输入
    input = [{'image': image_path}]

    # 调用API
    result = dashscope.MultiModalEmbedding.call(
        model=dashscope.MultiModalEmbedding.Models.multimodal_embedding_one_peace_v1,
        input=input,
        auto_truncation=True
    )

    print(result['output']['embedding'])
    print('output embedding dimension:', len(result['output']['embedding']))

if __name__ == '__main__':
    # 指定本地图片路径
    image_path = '/home/xiaodong.hu/tools/example.png'
    image_call(image_path)