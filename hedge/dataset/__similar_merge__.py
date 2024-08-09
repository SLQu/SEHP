import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Merging:
    
    def __init__(self,model_name = 'bert-base-chinese') -> None:
        # 加载预训练的BERT模型和分词器
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)

    # 将句子编码为BERT嵌入
    def encode_sentences(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=128, add_special_tokens=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings




    def calculate_similarity(self, query_embedding, sentence_embeddings):
        # query_embedding = query_embedding.unsqueeze(0)  
        similarity_scores = cosine_similarity(query_embedding.cpu().numpy(), sentence_embeddings.cpu().numpy())
        return similarity_scores[0]  

    # 合并相似的句子
    def merge_similar_sentences(self, sentences, threshold=0.95):

        num_sentences = len(sentences)
        
        merged_sentences = []
        sentence_mapping_oid_newid = {}
        
        for i in range(num_sentences):
            # print(f"Processing sentence {i + 1} of {num_sentences}, merged {len(merged_sentences)}.")
            if len(merged_sentences) == 0:
                merged_sentences.append(sentences[i])
                sentence_mapping_oid_newid[i] = 0
                sentence_embeddings = self.encode_sentences(sentences[i])
                continue
            
            else:
                query_embedding = self.encode_sentences(sentences[i])
                similarity_scores = self.calculate_similarity(query_embedding, sentence_embeddings)
                max_similarity = np.max(similarity_scores)
                
                if max_similarity > threshold:
                    max_similarity_index = np.argmax(similarity_scores)
                    sentence_mapping_oid_newid[i] = max_similarity_index
                    
                else:
                    sentence_mapping_oid_newid[i] = len(merged_sentences)
                    merged_sentences.append(sentences[i])
                    
                    sentence_embeddings = torch.cat((sentence_embeddings, query_embedding), 0)

        return merged_sentences, sentence_mapping_oid_newid
    
    def run(self,sentences,threshold=0.95):
        merged_sentences, sentence_mapping = self.merge_similar_sentences(sentences, threshold=threshold)
        return merged_sentences,sentence_mapping

if __name__ == "__main__":

    sentences = ['遵守规章制度，不要忘记关门', '遵守公司的规章制度，保持对工作的认真态度', '遵守社会公德，不影响他人生活和工作', '尊重他人的知识产权和隐私', '在正式场合应该用更加正式的语言和用词', '赞美他人时，不要使用夸张的用语', '尊重上级，听从指挥', '避免讨论敏感的话题', '倾听他人的需求并提供帮助', '礼仪：应该先礼后兵，开口前先问好', '在正式场合中，谈话应该遵循一定的规范和格式', '遵守办公室卫生标准，保持工作环境清洁整洁', '不随手乱扔垃圾，保持环境卫生', '以合理客观的理由支持自己的请求', '在晚上或早晨尽量不要制造噪音', '尊重他人的卫生和美观要求', '尊重服务员，礼貌用语', '设法使他人感到舒适和安心，不给他人造成压力或不安全感', '坚持高效周到负责任的工作态度', '不要过度承诺或大话，实事求是，讲真话', '不要以自己的主观看法去评价他人的美丑', '在公共交通工具和场所，要遵守秩序，避免推搡和打斗等不文明行为', '尽可能减少因语言沟通不畅而产生的误解', '遵守承诺，信守诺言', '发生错误要及时向对方道歉并解释', '尽快完成分配的任务，不要久拖不决', '如果需要批评他人，应该注重措辞和态度，避免让对方感到被攻击或冒犯', '吃早餐有助于维持身体健康和正常的消化功能', '遇到错误行为要及时道歉', '尽量不影响其他人的自由和权利', '避免谈及敏感话题，例如政治宗教等，以免引起争议和不和', '对陌生人应当保持一定的社交距离', '在决定购买商品时，可以表达感谢并询问一些具体的问题', '以合作为目的，争取共赢', '表达要求时，要有礼貌，不要过于强硬或咄咄逼人', '子女应该听从家长的安排，尊重家庭规矩', '有效沟通需要彼此理解和耐心', '表达意见时要尽量委婉', '当发现自己的行为可能影响到他人时，及时道歉并改正错误', '与上级交流时应该恰当地使用敬称', '在公共场合中保持安静和有秩序', '在正式场合或与上级交流时，应保持恰当的语气和言辞', '确保劳动环境健康和安全', '注意礼貌，说话不要粗鲁', '表达自己的意见和想法时，要注意用语和态度，避免冒犯他人', '避免在公共场合大声喧哗', '在社交场合中，要遵循社交礼仪和文化规范', '尊重他人，不要口出恶言', '约定时间和地点的确认', '在对话中尊重对方的意见和立场', '遵守合理的商业道德，不定价虚高或欺诈消费者', '尊重年长者和社会地位更高的人', '鼓励失业人员领取适当的救济金', '不应该因少量金钱而违反规定或道德', '给予鼓励和支持可以激发人的积极性和动力', '在公共场合保持文明礼貌', '避免给他人带来不必要的麻烦', '在造成他人损失时，应该承担相应的责任并且道歉', '不要太夸张', '适当使用感叹词和修辞手法，以表达感情加强语气或引起共鸣', '在困难和挑战面前保持坚韧和积极', '在处理问题时，要沉着冷静理智分析，不偏听偏信', '在办公场所中，应该保持安静和整洁，尤其是在上级或同事面前', '礼貌和尊重是每个人都应该具有的基本品质', '保持沟通和协作', '注意自己的举止和言行，表现出优秀的职业和个人素养', '对自己的行为负责，承担自己应承担的责任和义务', '尊重环境，保护地球', '不要对他人进行人身攻击和侮辱', '意识到自己的行为对他人有影响，保持社会责任感', '遵守社交礼仪，不要中断对方的话语', '不要轻易谈论敏感话题', '表现出相互尊重和友好的态度', '面对下属要委婉表达，不要直接指责', '在同伴关系中，应坦诚相待，保持友好互助的互动方式', '礼仪：遵守礼仪规范，例如始终用尊称称呼长辈或上级，行为得体，不失礼仪', '遵循习俗和传统，尊重文化差异', '不要让情绪控制自己的言行，保持冷静', '注意言行，不要伤害他人', '在公共场合应该注意自己的行为和言论，不要影响他人的安宁', '尊重别人的权利和利益是基本礼仪', '在商务交往中，需保持诚实守信，避免说谎或欺骗', '尊重隐私权和个人空间，不要窥探他人隐私', '认真落实工作要求，避免粗心马虎', '向上级或长辈请教问题时应先简单自我介绍', '事先询问别人的意见或需求有助于避免冲突', '充分准备和考虑可能出现的问题', '以友善和礼貌的方式沟通交流', '强调互惠互利的关系', '注意细节，注重细节体现职业态度', '在工作场所中要友善礼貌地与同事交流', '不应该因为领取失业救济金而歧视别人', '在受到他人指责时，要冷静回应，认真考虑是否存在自己的不当行为', '禁止使用粗言秽语或侮辱性话语，在社会交际中保持良好的言行举止', '繁忙场所避免大声喧哗，尊重他人的休息和工作', '了解维修费用和所需时间，有计划地安排好时间和预算', '如果要做出重大决定，应该寻求家庭的理解和谅解']
    merge = Merging()
    merged_sentences, sentence_mapping = merge.run(sentences, threshold=0.80)

    # 打印合并后的句子
    for i, sentence in enumerate(merged_sentences):
        print(f"{i + 1}: {sentence}")

    # 返回100与k之间的映射
    print(f"{len(sentences)}")
    print("Sentence Mapping:")
    print(sentence_mapping)
