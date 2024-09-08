import ast
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import re
import time

import csv
import sqlite3
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain_community.utilities.sql_database import SQLDatabase
import json
from langchain_groq import ChatGroq
import os
openai_api_key = ""
gorq_api_key = ""
os.environ["OPENAI_API_KEY"] = ""
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db_all = FAISS.load_local("FAISS_db/db_all", embeddings, allow_dangerous_deserialization=True)
main_prompt = '''
Định nghĩa:
    - Bạn là một chuyên gia tư vấn bán hàng và chăm sóc khách hàng của VCC
    - Bạn nói chuyện với phong thái lịch sự, cung cấp thông tin dễ đọc bằng cách liệt kê hoặc xuống dòng một cách logic và tự nhiên
    - Hãy xưng mình là em, khách hàng là anh/chị nếu chưa biết giới tính
    - Hãy luôn sử dụng từ ngữ viết tắt kèm các emoji vào cuộc hội thoại để tăng tính tự nhiên.
    - Bạn cũng từng trải nghiệm mua các sản phẩm trực tuyến và được tư vấn chăm sóc tận tình.
    Các kênh Liên hệ:
        Khi khách hàng có nhu cầu liên hệ với VCC thì thông tin liên hệ của VCC như sau:
        Hotline: 18009377
        e-mail: info.vccsmart@gmail.com
        website: https://aiosmart.com.vn/
        Địa chỉ: Số 6 Phạm Văn Bạch, P. Yên Hòa, Q. Cầu Giấy, Hà Nội
Mục tiêu:
    - Bán hàng, lấy thông tin khách hàng
    - Tư vấn chủ động giúp khách hàng hài lòng và giải quyết được các nhu cầu, vấn đề của khách hàng
    - Đảm bảo trải nghiệm mua hàng tốt nhất cho khách hàng
Nhiệm vụ:
  - Tư vấn, bán hàng và chốt đơn dựa vào dữ liệu sản phẩm để giải quyết những vấn đề của khách hàng
  - Chủ động xin thông tin của khách hàng
  - Chăm sóc và theo dõi tình trạng đơn hàng của khách hàng sau khi chốt đơn
  - Cần giao tiếp với khách hàng một cách, dễ hiểu, đi vào trọng tâm vấn đề; không cần quá sáng tạo hay bay bổng
  - Hãy liệt kê 3-5 sản phẩm có giá cao nhất nếu có thông tin trong Dữ liệu và sắp xếp theo giá từ cao đến thấp để gửi thông tin cho khách hàng và gọi đó là những sản phẩm cao cấp một cách tự nhiên để khách hàng có cảm giác thèm muốn.
Giới hạn:
  - Chỉ trả lời khách hàng bằng tiếng việt
  - Chỉ được phép sử dụng thông tin sản phẩm trong Dữ liệu
  - Không được phép gợi ý thêm sản phẩm không có trong Dữ liệu
  - Không được phép đưa thông tin giá cả không có trong Dữ liệu
  - Khách hàng cần độ chính xác 100%, nếu không có yêu cầu khác ngoài việc tư vấn sản phẩm công ty từ chối khách hàng một cách nhẹ nhàng
  - Nếu khách hàng viết tắt mà không hiểu rõ cần hỏi lại khách hàng, cấm không được bịa ra ý nghĩa của từ viết tắt.
  - Tuyệt đối không được giảm giá dưới bất kì hình thức nào, phải thuyết phục khách hàng đây là hàng xịn nên mới có giá đó
Quá trình bán hàng và chăm sóc khách hàng:
  - Bước 1: Hỏi về vấn đề hoặc nhu cầu của khách hàng
  - Bước 2: Dựa vào mô tả của sản phẩm để đề xuất sản phẩm và giải pháp để giải quyết các vấn đề và nhu cầu của khách hàng. Kết thúc mỗi lần tư vấn thì phải luôn kèm sau đó những lời gợi ý mua hàng, không được phép lặp lại quá 2 lần những câu này, hoặc thay đổi linh hoạt theo mẫu gợi ý sau:
        Mẫu gợi ý:
        “Anh/chị đặt mua sản phẩm về trải nghiệm nhé?”
        “Anh mua điều hòa về dùng thử nhé?”
  - Bước 3: Chốt đơn hàng thì cần cảm ơn khách hàng đã đặt hàng, tiếp theo đó là xác nhận bằng cách liệt kê lại tổng số sản phẩm khách đã đặt, kèm tên gọi và giá bán từng sản phẩm
    Ví dụ: Tuyệt vời, em xác nhận lại đơn hàng của mình gồm…giá…tổng đơn của mình là…”, rồi mới hỏi lại thông tin họ tên, sđt, địa chỉ nhận hàng của khách hàng.
    Tổng giá trị đơn hàng sẽ bằng giá sản phẩm * số lượng
Dữ liệu:
- Cửa hàng chỉ bán các loại sản phẩm: Bàn là, Bàn ủi, máy sấy tóc, bình nước nóng, bình đun nước, bếp từ, công tắc, ổ cắm, ghế massage daikosan, lò vi sóng, máy giặt, máy sấy, máy lọc không khí, máy lọc nước, máy xay, nồi chiên không dầu, nồi cơm điện, nồi áp suất, robot hút bụi, camera, webcam, thiết bị wifi, máy ép, tủ mát, quạt điều hòa không khí, máy làm sữa hạt, điều hòa, đèn năng lượng.
- {context}
- Dữ liệu ở trên là tri thức của bạn, hãy sử dụng các thông tin đó để hỗ trợ khách hàng
### Cuộc trò chuyện hiện tại:
{history}
Khách hàng:{question}
Shop:'''
def get_intent_entity(history, question):
    prompt_intent_entity = """Nhiệm vụ của bạn là phân tích câu hỏi người dùng kết hợp với lịch sử trò chuyện để trả ra output là json chứa entity và intent mà không kèm theo bất cứ lời giải thích nào.
    intent là ý định của người dùng, là một trong các kết quả sau:
    "Chào hỏi"(chào shop, chào em,..)
    "Cần tư vấn thông tin về các sản phẩm"(khi chưa rõ thông tin về tên, số lượng khách muốn mua)
    "Đồng ý mua hàng"(khi đã rõ tên, số lượng khách muốn mua)
    "Cần thay đổi đơn hàng"
    "Gửi thông tin cá nhân"
    entity là các sản phẩm liên quan đến câu hỏi của khách hàng, ví dụ: "bếp từ", "bàn là", "máy sấy", "máy xay", "nồi cơm", "nồi áp suất", "đèn năng lượng",...
    hoặc entity có thể là tên chính xác của sản phẩm nếu khách hàng muốn hỏi về sản phẩm cụ thể, ví dụ: "Máy sấy tóc 1200W Philips BHC010/10", "Ghế Massage Daikiosan DVGM-20001",...
    entity có thể có nhiều đối tượng, nếu có nhiều hãy trả ra tối đa 3 entity
    ---
    Chỉ trả về json chứa intent và entity
    Ví dụ:
    Khách hàng: xin chào
    output: \n  "intent": "Chào hỏi",\n  "entity": []\n
    Lịch sử trò chuyện:
    ---
    {history}
    ---
    Khách hàng: {question}
    ---
    output:
    """
    llm = ChatGroq(model='llama3-70b-8192', groq_api_key = "", temperature=0)
    json_data = llm.invoke(prompt_intent_entity.format(history = history, question = question)).content
    try:
        json_data = re.search(r'\{.*?\}', json_data, re.DOTALL).group()
    except:
        print('error in extract intent')
        json_data = '{\n"intent": "Cần tư vấn thông tin về các sản phẩm",\n"entity": []\n}'
    return json_data

csv_file_path = 'data/csv_file/204_new.csv'

# Kết nối đến cơ sở dữ liệu SQLite hoặc tạo một cơ sở dữ liệu mới
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Tạo bảng trong cơ sở dữ liệu
cursor.execute('''CREATE TABLE IF NOT EXISTS data_items (
                    LINK_SP TEXT,
                    PRODUCT_INFO_ID TEXT,
                    GROUP_PRODUCT_NAME TEXT,
                    PRODUCT_CODE TEXT,
                    PRODUCT_NAME TEXT,
                    SPECIFICATION_BACKUP TEXT,
                    RAW_PRICE INTEGER,
                    QUANTITY_SOLD
                    )''')

# Đọc dữ liệu từ tập tin CSV và chèn vào bảng
with open(csv_file_path, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Bỏ qua dòng tiêu đề nếu có
    for row in csv_reader:
        cursor.execute('''INSERT INTO data_items (LINK_SP, PRODUCT_INFO_ID, GROUP_PRODUCT_NAME, PRODUCT_CODE, PRODUCT_NAME, SPECIFICATION_BACKUP, RAW_PRICE, QUANTITY_SOLD)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', row)


# Lưu thay đổi và đóng kết nối
conn.commit()
conn.close()
from langchain_community.utilities.sql_database import SQLDatabase
db = SQLDatabase.from_uri("sqlite:///database.db",max_string_length=32000,)

def drop_columns(file_path):
    df = pd.read_csv(file_path)
    # df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    # Xóa các cột chỉ định
    columns_to_drop = ['Unnamed: 0', 'NON_VAT_PRICE_1', 'VAT_PRICE_1', 'COMMISSION_1', 'THRESHOLD_1', 'NON_VAT_PRICE_2',
                       'VAT_PRICE_2', 'COMMISSION_2', 'THRESHOLD_2', 'NON_VAT_PRICE_3', 'VAT_PRICE_3',
                       'COMMISSION_3']  # Thay bằng tên các cột bạn muốn xóa
    df = df.drop(columns=columns_to_drop)

    # Lưu tệp CSV đã được chỉnh sửa
    output_file_path = 'data/csv_file/204_new.csv'  # Thay bằng đường dẫn tới tệp CSV đầu ra của bạn
    df.to_csv(output_file_path, index=False)
# drop_columns('C:/Users/ADMIN/PycharmProjects/langchain_item_new/data/csv_file/product_final_204_ok.csv')
def convert_to_list_of_tuples(ccss):
    ccs = ast.literal_eval(ccss.strip())

    # Tạo danh sách các dòng text từ kết quả
    output = ""
    for row in ccs:
        product_name = row[0]  # Lấy giá trị tên sản phẩm từ phần tử đầu tiên của tuple
        raw_price = row[1]     # Lấy giá trị giá từ phần tử thứ hai của tuple
        sp = f"Tên sản phẩm: {product_name}, Giá: {raw_price}"
        output += sp.replace('\n',' ') + '\n'

    # Hiển thị kết quả
    return output
def convert_to_list_of_tuples_attribute(ccss, attributes):
    ccs = ast.literal_eval(ccss.strip())

    # Tạo danh sách các dòng text từ kết quả
    output = ""
    for row in ccs:
        product_name = row[0]  # Lấy giá trị tên sản phẩm từ phần tử đầu tiên của tuple
        raw_price = row[1]     # Lấy giá trị giá từ phần tử thứ hai của tuple
        backup = row[2]
        lines = backup.splitlines()
        attribute_txt =''
        # Lọc ra dòng chứa attribute
        for attribute in attributes:
            if attribute != 'giá':
                result_attribute = ''
                result = [line for line in lines if attribute in line.lower()]
                for line in result:
                    result_attribute = result_attribute + line
                attribute_txt = attribute_txt + result_attribute +' '
        sp = f"Tên sản phẩm: {product_name}, Giá: {raw_price}, {attribute_txt}"
        output += sp.replace('\n',' ') + '\n'

    # Hiển thị kết quả
    return output
def remove_duplicate_lines(input_string):
    # Tách chuỗi thành các dòng
    lines = input_string.splitlines()

    # Sử dụng set để loại bỏ các dòng lặp lại, giữ nguyên thứ tự xuất hiện
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)

    # Nối lại các dòng thành một chuỗi
    output_string = "\n".join(unique_lines)

    return output_string
def get_price(product_name,db):
    sql ='''SELECT RAW_PRICE FROM data_items WHERE PRODUCT_NAME LIKE "%{product_name}%" or GROUP_PRODUCT_NAME LIKE "%{product_name}%"'''
    ccs = db.run(sql.format(product_name=product_name))
    data = eval(ccs)

    # Lấy số từ tuple đầu tiên
    number = data[0][0]
    return number


def get_context(intent, entities, question,assistant, db):
    sql ='''SELECT PRODUCT_NAME, RAW_PRICE FROM data_items WHERE PRODUCT_NAME LIKE "%{entity}%" or GROUP_PRODUCT_NAME LIKE "%{entity}%" ORDER BY RAW_PRICE ASC'''
    sql_attribute = '''SELECT PRODUCT_NAME, RAW_PRICE, SPECIFICATION_BACKUP FROM data_items WHERE PRODUCT_NAME LIKE "%{entity}%" or GROUP_PRODUCT_NAME LIKE "%{entity}%" ORDER BY RAW_PRICE ASC'''
    context =''
    if intent =='Chào hỏi':
        context = ''
    # if intent =='Cần mua một loại sản phẩm dựa trên một bối cảnh cụ thể để phù hợp với nhu cầu':
    #     context = 'Khách hàng đang: '+ intent + 'Cửa hàng có bán các sản phẩm: Bàn là, Bàn ủi, máy sấy tóc, bình nước nóng, bình đun nước, bếp từ, công tắc ổ cắm, ghế massage daikosan, lò vi sóng, máy giặt, máy sấy, máy lọc không khí, máy lọc nước, máy xay, nồi chiên không dầu, nồi cơm điện, nồi áp suất, robot hút bụi, camera, webcam, thiết bị wifi, máy ép, tủ mát, quạt điều hòa không khí, máy làm sữa hạt, điều hòa, đèn năng lượng.'
    if intent =='Cần tư vấn thông tin về các sản phẩm':
        if entities != []:
            context ='Thông tin chi tiết một số sản phẩm:\n'
            dict_entity = []
            for entity in entities:
                relevant_documents = db_all.as_retriever(search_kwargs={"k": 2}).get_relevant_documents(entity)
                page_contents = [doc.page_content for doc in relevant_documents]
                context_mini = '\n'.join(page_contents)
                context = context + '\n' + context_mini
            context = context + '\n'
            context = context + 'Danh sách tên và giá một số sản phẩm sắp xếp tăng dần theo giá:\n'
            for entity in entities:
                entity = entity.lower()
                if 'ủi' in entity:
                    entity = 'bàn là'
                if 'lọc' in entity:
                    entity = 'lọc'
                if 'dầu' in entity or 'chiên' in entity:
                    entity = 'dầu'
                if 'nước' in entity:
                    entity = 'nước'
                if 'cơm' in entity:
                    entity = 'cơm'
                if 'quạt' in entity:
                    entity = 'quạt'
                if 'nồi' in entity:
                    entity = 'nồi'
                if 'cam' in entity:
                    entity = 'cam'
                if 'fi' in entity:
                    entity = 'fi'
                if 'hòa' in entity:
                    entity = 'hòa'
                if 'đèn' in entity:
                    entity = 'đèn'
                if 'điều' in entity:
                    entity = 'điều'
                if 'massa' in entity or 'mát xa' in entity or 'mát sa' in entity:
                    entity = 'massage'
                if 'giặt' in entity:
                    entity = 'giặt'
                if 'bụi' in entity:
                    entity = 'bụi'
                if 'xay' in entity:
                    entity = 'xay'
                if 'bếp' in entity:
                    entity ='bếp'
                if 'lò' in entity:
                    entity = 'lò'
                if entity in dict_entity:
                    print('_CUT_')
                    continue
                # print('__ĐÂY CHƯA LỖI__')
                dict_entity.append(entity)
                prompt = '''
                Nhiệm vụ:
                - Phân tích câu hỏi để trả ra một list thuộc tính mà khánh hàng muốn tìm kiếm
                - Thuộc tính có thể là:
                "màu"
                "công suất"
                "xuất xứ"
                "kích thước"
                "thương hiệu"
                ...
                - Bạn chỉ cần trả về một cách ngắn gọn là một đối tượng list
                - Không được phép giải thích bất cứ điều gì
                - Nếu không tìm được thuộc tính tương ứng, trả về: []
                Ví dụ:
                Khách hàng: có máy giặt lồng đứng không
                Output:['lồng']
                ---
                Khánh hàng: có máy xay nào dung tích dưới 5l không
                Output:['dung tích']
                ---
                Khánh hàng: có quạt điện nào 80W mà nhỏ hơn 2 triệu không
                Output:['công suất','giá']
                ---
                Khánh hàng: có máy pha cà phê không
                Output:[]
                ---
                Dựa vào các ví dụ trên, bạn hãy trả ra một list thuộc tính mà khách hàng muốn tìm kiếm:
                Khách hàng: {question}
                Output:
                '''
                llm = ChatGroq(model='llama3-70b-8192', groq_api_key = "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", temperature=0)
                attributes = llm.invoke(prompt.format(question = question)).content
                # string = ".....[...]...."
                print('attributes:', attributes)
                try:
                    attributes = re.search(r'\[.*?\]', attributes, re.DOTALL).group()
                except:
                    print('error in attributes')
                    attributes ='[]'
                attributes = ast.literal_eval(attributes)
                print('attributes:', attributes)
                if attributes != []:
                    print('____USING ATTRIBUTE____')
                    sql1 = sql_attribute.format(entity=entity)
                    print('SQL:', sql1)
                    ccs = db.run(sql1)
                    print('SQL result:', ccs)
                    try:
                        context = context + convert_to_list_of_tuples_attribute(ccs, attributes) +'\n'
                    except:
                        print('error in ccs attributes')
                        # continue
                else:
                    print('____USING PRICE____')
                    sql1 = sql.format(entity=entity)
                    ccs = db.run(sql1)
                    print('ccs:',ccs)
                    try:
                        context = context + convert_to_list_of_tuples(ccs) +'\n'
                    except:
                        print('error in ccs')
                        # continue
        # else:
        # context = 'Cửa hàng có bán các sản phẩm: Bàn là, Bàn ủi, máy sấy tóc, bình nước nóng, bình đun nước, bếp từ, công tắc ổ cắm, ghế massage daikosan, lò vi sóng, máy giặt, máy sấy, máy lọc không khí, máy lọc nước, máy xay, nồi chiên không dầu, nồi cơm điện, nồi áp suất, robot hút bụi, camera, webcam, thiết bị wifi, máy ép, tủ mát, quạt điều hòa không khí, máy làm sữa hạt, điều hòa, đèn năng lượng.'
    if intent =='Muốn giảm giá':
        context = '- Không được giảm giá dưới bất kì hình thức nào, phải bảo khách hàng đây là hàng xịn nên mới có giá đó'
    if intent == 'Đồng ý mua hàng':
        # Mở tệp 'danh sách sản phẩm.txt' để ghi thêm vào cuối tệp
        with open('ds_sp.txt', 'a', encoding='utf-8') as file:
            shop = 'Shop:' + assistant+'\n' + 'Khách hàng:' + question+'\n---\n'
            file.write(shop)
            # file.write()
            # for entity in entities: Xử lý được 1 sản phẩm
            #     # Ghi nội dung vào tệp
            #     price_txt =''
            #     try:
            #         price = get_price(entity)
            #         price_txt = ', Giá: '+ str(price)
            #     except:
            #         price_txt =''
            #         print('error in get price')
            #     file.write(entity+price_txt+'\n')  # Thay thế bằng nội dung bạn muốn ghi thêm

        # Mở lại tệp 'danh sách sản phẩm.txt' để đọc toàn bộ nội dung
        with open('ds_sp.txt', 'r', encoding='utf-8') as file:
            content = file.read()

        # In toàn bộ nội dung của tệp ra màn hình
        context = 'Đọc các đoạn chat để biết khách hàng đã chốt mua các sản phẩm nào:\n' + content +'Hãy xác nhận với khách đơn hàng nếu đã có thông tin địa chỉ của khách hàng, nếu chưa có thông tin của khách thì hỏi lại khách thông tin tên, số điện thoại, địa chỉ'
    if intent =='Gửi thông tin cá nhân':
        context = '- Hãy viết lại đơn hàng bao gồm thông tin cá nhân của khách kèm theo sản phẩm khách mua'
        with open('tt_cn.txt', 'w', encoding='utf-8') as file:
            file.write(question)  # Thay thế bằng nội dung bạn muốn ghi thêm
    context = remove_duplicate_lines(context)
    print('intent:',intent)
    print('entities:',entities)
    print('__context__:',context)
    if len(context) > 12000:
        print('____WARNING____')
    return context[:12000]
def build_history(history):
    if len(history) > 3:
        short_history = history[-3:]
    else:
        short_history = history
    history_str = ""
    for human, assistant in short_history:
        history_str += f"Khách hàng: {human}\nShop: {assistant}\n"
    return history_str
def predict(message, history):
    # print(history)
    history_openai_format = []
    if len(history) == 0:
      for human, assistant in history:
          history_openai_format.append({"role": "user", "content": human })
          history_openai_format.append({"role": "assistant", "content": assistant})
    #   docs = vectorstore.similarity_search(message)
    #   context = docs[0].page_content
      json_data = get_intent_entity(build_history(history), message)
      json_dict = json.loads(json_data)
      print('Khách:',message)
      entities = json_dict["entity"]
      intent = json_dict["intent"]
      context = get_context(intent, entities,message,'',db)
      history_openai_format.append({"role": "user", "content": main_prompt.format(context=context, question=message, history='')})
    else:
      history_str = build_history(history)
    #   print(history_str)
    #   docs = vectorstore.similarity_search(history_str)
    #   context = docs[0].page_content
      json_data = get_intent_entity(history_str, message)
      json_dict = json.loads(json_data)
      print('Khách:',message)
      entities = json_dict["entity"]
      intent = json_dict["intent"]
      human, assistant = history[-1]
      context = get_context(intent, entities,message,assistant,db)
      if os.path.exists('tt_cn.txt'):
        # Nếu tệp tồn tại, mở và đọc nội dung của tệp
        with open('tt_cn.txt', 'r', encoding='utf-8') as file:
            content = file.read()
        context = context + '\nĐây là thông tin tên, số điện thoại, địa chỉ của khách hàng:\n' + content+'\n'
        # os.remove('tt_cn.txt')
      history_openai_format.append({"role": "user", "content": main_prompt.format(context=context, question=message, history=history_str) })

    print(history_openai_format)
    response = ChatGroq(api_key="",
      model='llama3-70b-8192',
    #   messages= ,
      temperature=0,
    #   streaming=True
                                           ).stream(history_openai_format)

    partial_message = ""
    for chunk in response:
        # print(chunk)
        if chunk.content is not None:
              partial_message = partial_message + chunk.content
              yield partial_message



# gr.ChatInterface(predict).launch(debug=True, share=True)
# if intent =='Muốn giảm giá':
#     answer = llm.invoke(question).content
# if intent =='Đồng ý mua hàng':
#     answer = llm.invoke(question).content
# if intent =='Gửi thông tin cá nhân':
#     answer = llm.invoke(question).content
