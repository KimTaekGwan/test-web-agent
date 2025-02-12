import json
import openai
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser


############################
# 1) 에멧 파서 (간단 버전)
############################
class TagNode:
    def __init__(self, tag: str):
        self.tag = tag
        self.children: List["TagNode"] = []

    def to_html(self, text_map: dict, indent: int = 0) -> str:
        """
        text_map: {id(self_node): "LLM가 생성한 텍스트"} 형태
        """
        indent_str = "  " * indent
        opening = f"{indent_str}<{self.tag}>"
        closing = f"{indent_str}</{self.tag}>"
        node_text = text_map.get(id(self), "")

        if not self.children:
            # 자식 없으면 한 줄
            return f"{opening}{node_text}{closing}"
        else:
            # 자식 있으면 줄바꿈 처리
            lines = []
            if node_text:
                lines.append(f"{indent_str}  {node_text}")
            for child in self.children:
                lines.append(child.to_html(text_map, indent + 1))

            inner_html = "\n".join(lines)
            return f"{opening}\n{inner_html}\n{closing}"


class EmmetParser:
    """
    >, +, *, () 정도만 처리하는 간단 파서
    """

    def __init__(self, expression: str):
        self.expr = expression.strip()
        self.len_expr = len(self.expr)
        self.index = 0

    def current_char(self):
        return self.expr[self.index] if self.index < self.len_expr else ""

    def eat_char(self):
        ch = self.current_char()
        self.index += 1
        return ch

    def parse_sequence(self) -> List[TagNode]:
        """
        최상위에서 '+' 로 형제 구분
        """
        nodes = []
        while self.index < self.len_expr:
            item_nodes = self.parse_item()
            nodes.extend(item_nodes)
            if self.current_char() == "+":
                self.eat_char()  # '+' 소비
            else:
                break
        return nodes

    def parse_item(self) -> List[TagNode]:
        """
        괄호 or 태그 -> *N 반복 -> > 자식
        """
        if self.current_char() == "(":
            self.eat_char()  # '('
            sub_nodes = self.parse_sequence()
            if self.current_char() == ")":
                self.eat_char()  # ')'
            else:
                raise ValueError("괄호가 제대로 닫히지 않음.")
        else:
            # 태그명
            tag_name = self.parse_tag_name()
            sub_nodes = [TagNode(tag_name)]

        # *N
        sub_nodes = self.parse_repetition(sub_nodes)

        # > 자식
        while self.current_char() == ">":
            self.eat_char()  # '>'
            child_nodes = self.parse_sequence()
            for parent_node in sub_nodes:
                for c in child_nodes:
                    parent_node.children.append(self.clone_tree(c))

        return sub_nodes

    def parse_tag_name(self) -> str:
        start = self.index
        while self.current_char().isalnum():
            self.eat_char()
        tag_name = self.expr[start : self.index]
        if not tag_name:
            raise ValueError("태그명 파싱 오류")
        return tag_name

    def parse_repetition(self, nodes: List[TagNode]) -> List[TagNode]:
        if self.current_char() == "*":
            self.eat_char()  # '*'
            num_str = ""
            while self.current_char().isdigit():
                num_str += self.eat_char()
            if not num_str:
                raise ValueError("* 뒤에는 숫자가 와야 함.")
            count = int(num_str)
            results = []
            for _ in range(count):
                for n in nodes:
                    results.append(self.clone_tree(n))
            return results
        return nodes

    def clone_tree(self, node: TagNode) -> TagNode:
        new_n = TagNode(node.tag)
        for c in node.children:
            new_n.children.append(self.clone_tree(c))
        return new_n


def parse_emmet(emmet_code: str) -> List[TagNode]:
    parser = EmmetParser(emmet_code)
    return parser.parse_sequence()


# 기존 TagContent (HTML 태그와 텍스트 검증 모델) - self-reference 처리를 위해 update_forward_refs 사용
class TagContent(BaseModel):
    tag: str = Field(..., description="HTML 태그 이름 (예: div, h1, p 등)")
    text: Optional[str] = Field(default="", description="태그 내부의 텍스트 내용")
    children: List["TagContent"] = Field(
        default_factory=list, description="하위 태그들의 목록"
    )

    @classmethod
    def check_text_len(cls, v, info):
        tag = info.data.get("tag", "")
        limit_map = {
            "h1": 20,
            "p": 100,
            "div": 200,
            "span": 50,
            "li": 30,
            "*": 50,
        }
        limit = limit_map.get(tag, limit_map["*"])
        if len(v) > limit:
            raise ValueError(f"<{tag}> 텍스트는 최대 {limit}자 이내여야 합니다.")
        return v


# self-reference 해결
TagContent.model_rebuild()


# LLM의 출력 전체 스키마 (최상위에 tagContents 배열)
class GenerateContentsOutput(BaseModel):
    tagContents: List[TagContent]


def build_tag_tree_json(nodes: List[TagNode]) -> List[dict]:
    # TagNode 트리를 LLM용 JSON 구조로 변환 (text 필드는 ""로 초기화)
    results = []
    for n in nodes:
        child_json = build_tag_tree_json(n.children)
        results.append({"tag": n.tag, "text": "", "children": child_json})
    return results


def build_text_map(nodes: List[TagNode], contents: List[TagContent]) -> dict:
    # TagNode와 TagContent가 동일한 구조/순서를 가진다고 가정하고 고유 id로 매핑
    text_map = {}
    for node, c in zip(nodes, contents):
        text_map[id(node)] = c.text
        if node.children and c.children:
            child_map = build_text_map(node.children, c.children)
            text_map.update(child_map)
    return text_map


## with_structured_output 사용 구현 예시


def generate_texts_for_emmet(emmet_code: str) -> str:
    # 1) 에멧 코드 파싱하여 TagNode 트리 생성
    root_nodes = parse_emmet(emmet_code)  # List[TagNode]

    # 2) LLM에 전달할 태그 트리 JSON 데이터 구성 (text는 공란 처리)
    tag_tree_data = build_tag_tree_json(root_nodes)

    # 3) LLM에게 줄 프롬프트 작성
    prompt = f"""
당신은 HTML 문서의 텍스트를 생성하는 어시스턴트입니다.
주어진 태그 구조에 맞춰, 각 태그에 들어갈 텍스트를 작성하고 JSON 형식으로 반환하세요.
각 태그의 글자 수 제한에 유의하여 적절한 텍스트를 생성해 주세요.
에멧 코드: {emmet_code}
""".strip()
    # 태그 트리: {json.dumps({"tagTree": tag_tree_data}, ensure_ascii=False)}

    # 4) Chat 모델에 structured output 스키마 바인딩 (즉, 직접 Pydantic 객체를 반환)
    structured_llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0.7
    ).with_structured_output(GenerateContentsOutput)

    # 5) 모델 호출 및 출력 객체 받기
    try:
        result: GenerateContentsOutput = structured_llm.invoke(prompt)
    except ValidationError as ve:
        # 스키마 검증 실패 시 예외 처리
        print("[Pydantic 검증 실패]", ve)
        raise ve
    except Exception as e:
        print("[오류 발생]", e)
        raise e

    # 6) 반환받은 tagContents를 기존 TagNode 트리와 매핑하여 HTML 텍스트 생성
    text_map = build_text_map(root_nodes, result.tagContents)
    final_html_list = [node.to_html(text_map) for node in root_nodes]
    return "\n".join(final_html_list)


# 테스트 (예시)
if __name__ == "__main__":
    test_emmet_codes = ["div>(h1+p)*2", "(h1+p)*3", "ul>li*5", "div>h1+span"]
    emmet_examples = [
        # 1. 기본 구조
        "div>h1+p",
        "ul>li*5",
        "nav>ul>li*3",
        "section>h2+p+button",
        "article>h3+p+span",
        # 2. 반복(*) & 그룹화(())
        "div>(h1+p)*3",
        "table>tr*3>td*4",
        "ul>(li>a)*5",
        "(header>h1)+(main>p*2)+(footer>p)",
        "div>(section>h2+p)*2",
        # 3. 중첩 구조
        "div>(header>h1+nav>ul>li*3)",
        "main>(section>h2+p)*3",
        "form>(fieldset>legend+input+button)*2",
        "article>(h2+p+span)*2",
        "div>(aside>h3+p)+(section>h2+p*2)",
        # 4. 복잡한 계층 구조
        "div>(header>h1)+(main>section*2>h2+p)",
        "section>(div>h2+p)+(div>h2+p+ul>li*3)",
        "footer>(div>p*2)+(nav>ul>li*4)",
        "div>(header>nav>ul>li*3)+(section>h2+p*2)+(footer>p)",
        "div>(header>h1)+(section>article*2>h2+p)+(footer>p)",
    ]

    # for code in test_emmet_codes:
    for code in emmet_examples:
        print("=== 에멧 코드:", code)
        try:
            html_output = generate_texts_for_emmet(code)
            print("[생성된 HTML]\n", html_output)
        except Exception as e:
            print("[오류]\n", e)
        print("-------------------------------------------------\n")
