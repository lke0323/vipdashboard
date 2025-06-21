import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import plotly.express as px
import pickle # 모델 저장/로드에 사용

# implicit 라이브러리 임포트 (설치 필수: pip install implicit)
import implicit
from scipy.sparse import csr_matrix # 희소 행렬 사용

# Windows에서 한글 깨짐 방지 (Mac 사용자는 'AppleGothic'으로 변경)
plt.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

# Streamlit 페이지 설정
st.set_page_config(page_title="InstaCart VIP 분석", layout="wide")

# 대시보드 제목 및 설명
st.title("🚀 InstaCart VIP 고객 전환 전략 및 행동 분석 대시보드")
st.markdown("""
이 대시보드는 1등급 고객 행동 분석을 기반으로,  
2~3등급 고객을 1등급으로 전환하기 위한 전략과 맞춤 추천을 제공합니다.
""")

# --- 데이터 로드 (캐시) ---
@st.cache_data(show_spinner=False)
def load_data():
    # 데이터 파일의 절대 경로를 지정합니다. (본인의 실제 경로에 맞게 수정 필요)
    # 예시: base_data_path = "C:/Users/SAMSUNG/Downloads/archive/data_InstaCart/"
    # 이 부분은 사용자 환경에 맞게 반드시 수정해야 합니다.
    base_data_path = "C:/Users/SAMSUNG/Downloads/archive/data_InstaCart/" # 실제 경로로 수정하세요!

    vip_df = pd.read_csv(f"{base_data_path}vip_summary_v2.csv")
    products = pd.read_csv(f"{base_data_path}products.csv")
    orders = pd.read_csv(f"{base_data_path}orders.csv")
    order_products_prior = pd.read_csv(f"{base_data_path}order_products__prior.csv")

    # 핵심: order_products_prior 에 user_id 와 product_name을 미리 병합하여
    # 이후의 모든 분석에서 사용할 통합 DataFrame을 만듭니다.
    combined_order_products = order_products_prior.merge(orders[['order_id', 'user_id']], on='order_id', how='left')
    combined_order_products = combined_order_products.merge(products[['product_id', 'product_name']], on='product_id', how='left')

    return vip_df, products, orders, combined_order_products

# --- implicit ALS 모델 학습 및 로드 ---
@st.cache_resource(show_spinner=False) # 모델은 한 번만 로드/학습되도록 cache_resource 사용
def train_and_load_als_model(combined_order_products, products_df): # products_df 인자 추가
    with st.spinner("🚀 추천 모델 학습 중... (데이터 양에 따라 시간이 오래 걸릴 수 있습니다)"):
        # 1. 사용자-아이템 매핑 생성
        unique_users = combined_order_products['user_id'].unique()
        unique_products = combined_order_products['product_id'].unique()

        user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        product_to_idx = {product_id: idx for idx, product_id in enumerate(unique_products)}

        # 역 매핑 (추천 결과를 원래 ID로 변환하기 위함)
        idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
        idx_to_product = {idx: product_id for product_id, idx in product_to_idx.items()}

        # 2. 상호작용 행렬 (희소 행렬) 생성
        num_users = len(unique_users)
        num_products = len(unique_products)

        data = np.ones(len(combined_order_products))
        rows = combined_order_products['user_id'].map(user_to_idx).astype(int)
        cols = combined_order_products['product_id'].map(product_to_idx).astype(int)

        user_item_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_products))

        # 3. ALS 모델 학습
        model = implicit.als.AlternatingLeastSquares(
            factors=64, # 잠재 요인 수
            regularization=0.05, # 과적합 방지 정규화 강도
            iterations=50, # 학습 반복 횟수
            random_state=42 # 재현성을 위한 랜덤 시드
        )

        model.fit(user_item_matrix)

    return model, user_to_idx, product_to_idx, idx_to_user, idx_to_product, user_item_matrix

# --- 추천 함수 (ALS 모델 사용) ---
@st.cache_data(show_spinner=False)
def recommend_products_als(_user_id, _model, _user_to_idx, _idx_to_product, _user_item_matrix, _products_df, N=5):
    if _user_id not in _user_to_idx:
        return []

    user_idx = _user_to_idx[_user_id]

    recommendations, scores = _model.recommend(
        user_idx,
        _user_item_matrix[user_idx],
        N=N,
        filter_already_liked_items=True
    )

    recommended_product_ids = [_idx_to_product[rec_idx] for rec_idx in recommendations]

    # 추천된 상품의 product_id와 product_name을 담은 DataFrame 반환
    recommended_products_info = _products_df[_products_df['product_id'].isin(recommended_product_ids)][['product_id', 'product_name']].copy()

    # 여기서 가상의 가격, 할인율, 리뷰 정보를 추가할 수 있습니다.
    # 실제 데이터가 있다면 해당 데이터프레임에서 merge하여 사용합니다.
    recommended_products_info['discount_rate'] = np.random.randint(5, 30, size=len(recommended_products_info))
    recommended_products_info['original_price'] = np.random.randint(10000, 50000, size=len(recommended_products_info))
    recommended_products_info['current_price'] = recommended_products_info['original_price'] * (1 - recommended_products_info['discount_rate'] / 100)
    recommended_products_info['review_score'] = np.round(np.random.uniform(3.5, 5.0, size=len(recommended_products_info)), 1)
    recommended_products_info['review_count'] = np.random.randint(5, 100, size=len(recommended_products_info))

    return recommended_products_info.to_dict('records') # 딕셔너리 리스트로 반환


# --- 메인 실행 흐름 ---
# 데이터 로드
vip_df, products, orders, combined_order_products = load_data()

# 모델 학습 및 로드 (products DataFrame을 인자로 전달)
model_als, user_to_idx, product_to_idx, idx_to_user, idx_to_product, user_item_matrix = train_and_load_als_model(combined_order_products, products)

# VIP 등급 부여
bins = [-0.1, 60, 70, 80, 90, 100]
labels = ['5.Bronze','4.Silver','3.Gold','2.Platinum','1.Diamond']
vip_df['vip_grade'] = pd.cut(vip_df['vip_score'], bins=bins, labels=labels)

# --- 전략 탭용 데이터 준비 함수 ---
@st.cache_data(show_spinner=False)
def prepare_strategy_data(vip_df, orders, combined_order_products):
    product_diversity = combined_order_products.groupby('user_id')['product_id'].nunique().rename('unique_product_count')

    vip_df = vip_df.set_index('user_id')
    vip_df = vip_df.join(product_diversity)
    vip_df.reset_index(inplace=True)

    group1 = vip_df[vip_df['vip_grade'] == '1.Diamond'].copy()
    group2 = vip_df[vip_df['vip_grade'].isin(['2.Platinum', '3.Gold'])].copy()
    group1['group'] = '1등급'
    group2['group'] = '2~3등급'
    compare_df = pd.concat([group1, group2])

    diamond_users = vip_df[vip_df['vip_grade'] == '1.Diamond']['user_id'].unique()
    diamond_orders = orders[orders['user_id'].isin(diamond_users)].copy()
    diamond_orders.sort_values(by=['user_id', 'order_number'], inplace=True)
    diamond_orders['days_since_prior_order'] = diamond_orders['days_since_prior_order'].fillna(0)
    avg_interval_df = diamond_orders.groupby('user_id')['days_since_prior_order'].mean().reset_index()
    avg_interval_df.rename(columns={'days_since_prior_order': 'avg_reorder_interval'}, inplace=True)

    return compare_df, avg_interval_df

compare_df, avg_interval_df = prepare_strategy_data(vip_df, orders, combined_order_products)

# Streamlit session_state 초기화 (장바구니)
if 'cart' not in st.session_state:
    st.session_state.cart = []

def add_to_cart(product_name, price):
    st.session_state.cart.append({'name': product_name, 'price': price})
    st.toast(f"'{product_name}'이(가) 장바구니에 담겼습니다! 🛒")


# --- Streamlit 탭 구성 ---
탭_개요, 탭_등급, 탭_1등급, 탭_전략, 탭_추천 = st.tabs([
    "🏠 개요",
    "📊 등급별 고객 분석",
    "🔎 1등급 고객 집중 분석",
    "💡 2~3등급 전환 전략",
    "🎯 맞춤형 추천 시스템"
])

# 각 탭의 내용 (이전 코드와 동일하게 유지)
with 탭_개요:
    st.header("🚀 InstaCart VIP 고객 분석 개요")
    st.markdown("""
    고객의 활동, 재구매, 최근성 등을 반영한 **VIP 스코어 기반의 등급 분류**를 통해,
    상위 고객군을 집중 분석하고, 2~3등급 고객을 1등급으로 전환하기 위한 전략과 맞춤 추천을 제공합니다.
    """)

    grade_counts = vip_df['vip_grade'].value_counts(normalize=True).reindex(labels).fillna(0)
    grade_percents = (grade_counts * 100).round(1)

    st.subheader("💡 고객 등급별 분포 현황")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("💎 1.Diamond", f"{grade_percents['1.Diamond']}%", help="최상위 VIP 고객 그룹입니다.")
    with col2:
        st.metric("💍 2.Platinum", f"{grade_percents['2.Platinum']}%", help="VIP 전환 가능성이 높은 고객 그룹입니다.")
    with col3:
        st.metric("🥇 3.Gold", f"{grade_percents['3.Gold']}%", help="성장 잠재력이 있는 고객 그룹입니다.")
    with col4:
        st.metric("🥈 4.Silver", f"{grade_percents['4.Silver']}%", help="기본적인 구매 활동을 하는 고객 그룹입니다.")
    with col5:
        st.metric("🥉 5.Bronze", f"{grade_percents['5.Bronze']}%", help="초기 또는 비활동 고객 그룹입니다.")

    st.markdown("---")

    st.subheader("📊 전체 고객 요약")
    col_total1, col_total2, col_total3 = st.columns(3)
    with col_total1:
        st.metric("👥 전체 고객 수", f"{vip_df.shape[0]:,}명")
    with col_total2:
        st.metric("👑 1등급 고객 비율", f"{grade_percents['1.Diamond']}%")
    with col_total3:
        st.metric("⭐ 평균 VIP Score", f"{vip_df['vip_score'].mean():.2f}점")


with 탭_등급:
    st.header("📊 고객 등급 분포 및 행동 패턴 비교")
    st.markdown("고객 등급 분포를 파악하고, 각 등급별로 어떤 행동 특성을 보이는지 비교합니다.")

    grade_counts = vip_df['vip_grade'].value_counts().reindex(labels)
    fig = px.bar(
        x=grade_counts.index,
        y=grade_counts.values,
        color=grade_counts.index,
        color_discrete_map={
            '1.Diamond': '#FFD700', # Gold
            '2.Platinum': '#C0C0C0', # Silver
            '3.Gold': '#CD7F32', # Bronze
            '4.Silver': '#ADD8E6', # LightBlue
            '5.Bronze': '#A9A9A9'  # DarkGray
        },
        labels={'x': 'VIP 등급', 'y': '고객 수'},
        title='✨ 고객 등급별 분포',
        template='plotly_white'
    )
    fig.update_traces(marker_line_color='black', marker_line_width=1.5)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown("""
    **분석 인사이트**: Instacart 고객의 대부분은 3~5등급에 분포해 있으며,
    상위 1~2등급 고객은 소수이지만 핵심적인 구매 활동을 수행합니다.
    """)


    with st.expander("🔍 1등급과 2~3등급 고객 행동 비교"):
        st.markdown(
            "1등급 고객과 2~3등급 고객의 주요 행동 지표를 비교합니다. "
            "이 비교는 **2~3등급 고객을 1등급으로 전환하기 위한 핵심 전략** 수립에 중요한 기반이 됩니다."
        )

        cols = [
            'total_orders', 'total_products', 'reorder_rate',
            'avg_cart_size', 'recency', 'unique_product_count'
        ]
        col_names = [
            '총 주문 수', '총 제품 수', '재구매율',
            '평균 장바구니 크기', '최근 구매 주기(일)', '고유 상품 수'
        ]

        fig2, axes = plt.subplots(2, 3, figsize=(18, 10)) # figsize 조정
        for i, (col, name) in enumerate(zip(cols, col_names)):
            ax = axes[i // 3, i % 3]
            sns.boxplot(x='group', y=col, data=compare_df, palette='viridis', ax=ax) # palette 변경
            ax.set_title(f"그룹별 {name} 비교", fontsize=14)
            ax.set_ylabel(name, fontsize=12)
            ax.set_xlabel("고객 그룹", fontsize=12)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)

        fig2.delaxes(axes[1][2]) # 빈 서브플롯 제거
        plt.tight_layout() # 레이아웃 자동 조정
        st.pyplot(fig2)

        st.markdown("#### 📝 평균값 요약표")
        st.markdown("각 고객 그룹별 주요 지표의 평균값을 한눈에 비교해 보세요.")
        group_means = (
            compare_df
            .groupby('group')[cols]
            .mean()
            .rename(columns=dict(zip(cols, col_names)))
            .round(2)
        )
        st.dataframe(group_means)
        st.markdown("""
        **주요 관찰**: 1등급 고객은 모든 지표에서 2~3등급 고객보다 우수한 수치를 보입니다.
        특히 **총 주문 수, 총 제품 수, 재구매율, 고유 상품 수**에서 큰 차이를 보여,
        이러한 지표들을 개선하는 것이 2~3등급 고객 전환의 핵심임을 시사합니다.
        """)


with 탭_1등급:
    st.header("🔎 1등급 고객 집중 행동 분석")
    st.markdown("InstaCart의 최상위 VIP 고객인 1등급 고객의 심층적인 행동 패턴을 분석하여 성공 요인을 파악합니다.")

    top_orders_data = combined_order_products.merge(
        orders[['order_id', 'order_number', 'order_dow', 'order_hour_of_day']],
        on='order_id', how='left'
    )
    top_users = vip_df[vip_df['vip_grade'] == '1.Diamond']
    top_orders = top_orders_data[top_orders_data['user_id'].isin(top_users['user_id'])]

    with st.expander("🛒 상위 구매 상품 분석"):
        st.markdown("1등급 고객이 가장 많이 구매하는 상품들을 통해 선호하는 카테고리나 상품 유형을 파악합니다.")
        top_products = top_orders['product_name'].value_counts().head(10).reset_index()
        top_products.columns = ['product_name', 'count']

        fig3 = px.bar(top_products, x='count', y='product_name', orientation='h',
                      title='🏆 1등급 고객 Top 10 구매 상품',
                      labels={'count': '구매 횟수', 'product_name': '상품명'},
                      color='count', color_continuous_scale=px.colors.sequential.Viridis)
        fig3.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""
        **핵심 인사이트**: 1등급 고객은 특정 **스테이플(staple) 상품**에 대한 높은 충성도와 반복 구매 경향을 보입니다.
        이는 이들 고객에게 **개인화된 추천 및 정기 배송 프로모션**이 효과적일 수 있음을 시사합니다.
        """)

    with st.expander("⏰ 활동 요일 및 시간대 분석"):
        st.markdown("1등급 고객의 주문 활동이 활발한 요일과 시간대를 분석하여 마케팅 메시지 전달의 최적 시점을 찾습니다.")
        col1, col2 = st.columns(2)
        with col1:
            dow_counts = top_orders['order_dow'].value_counts().sort_index().reset_index()
            dow_counts.columns = ['order_dow', 'count']
            fig4 = px.bar(dow_counts, x='order_dow', y='count',
                          title='📅 요일별 주문 분포 (0=일요일)',
                          labels={'order_dow': '요일', 'count': '주문 수'},
                          color='count', color_continuous_scale=px.colors.sequential.Blues)
            st.plotly_chart(fig4, use_container_width=True)
        with col2:
            hour_counts = top_orders['order_hour_of_day'].value_counts().sort_index().reset_index()
            hour_counts.columns = ['order_hour_of_day', 'count']
            fig5 = px.bar(hour_counts, x='order_hour_of_day', y='count',
                          title='⏱️ 시간대별 주문 분포',
                          labels={'order_hour_of_day': '시간대', 'count': '주문 수'},
                          color='count', color_continuous_scale=px.colors.sequential.Greens)
            st.plotly_chart(fig5, use_container_width=True)
        st.markdown("""
        **핵심 인사이트**: 1등급 고객의 활동은 특정 요일과 시간대에 집중되는 경향이 뚜렷합니다.
        이는 **타이밍에 맞춰 발송되는 푸시 알림, 이메일, 프로모션 메시지**의 효과를 극대화할 수 있음을 의미합니다.
        """)

    with st.expander("🔄 재구매 주기 분석"):
        st.markdown("1등급 고객의 평균 재구매 주기를 파악하여, 재주문 유도를 위한 최적의 시점을 결정합니다.")
        fig6 = px.histogram(avg_interval_df, x='avg_reorder_interval', nbins=30,
                            title='📈 1등급 고객 평균 재구매 주기 분포',
                            labels={'avg_reorder_interval': '평균 재구매 주기 (일)', 'count': '고객 수'},
                            color_discrete_sequence=['steelblue'],
                            template='plotly_white')
        fig6.update_traces(marker_line_color='black', marker_line_width=1.5)
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown(f"**평균 재구매 주기**: `{avg_interval_df['avg_reorder_interval'].mean():.2f}` 일")
        st.markdown(f"**중앙값 재구매 주기**: `{avg_interval_df['avg_reorder_interval'].median():.2f}` 일")
        st.markdown("""
        **핵심 인사이트**: 1등급 고객의 대부분은 **5~15일 사이**에 재구매를 수행합니다.
        이러한 **일정한 구매 주기**를 활용하여, 재구매 시점에 맞춰 **개인화된 리마인드 메시지**나
        **정기배송 서비스 유도 프로모션**을 제공하면 전환율을 높일 수 있습니다.
        """)

with 탭_전략:
    st.header("💡 2~3등급 → 1등급 전환 전략 상세 분석")
    st.markdown("1등급 고객과 2~3등급 고객 간의 행동 차이를 기반으로, 2~3등급 고객을 1등급으로 전환하기 위한 구체적인 전략을 제안합니다.")

    st.markdown("---")

    with st.expander("1️⃣ **다양한 상품 경험 유도 전략**"):
        st.markdown("""
        **문제 인식**: 1등급 고객은 2~3등급 고객보다 훨씬 **다양한 종류의 상품**을 구매합니다.
        **전략 목표**: 2~3등급 고객이 InstaCart에서 더 많은 상품 카테고리를 탐색하고 구매하도록 유도합니다.
        """)
        fig_diversity = px.bar(compare_df.groupby('group')['unique_product_count'].mean().reset_index(),
                     x='unique_product_count', y='group', orientation='h', color='group',
                     labels={'unique_product_count':'평균 고유 상품 수', 'group':'고객 그룹'},
                     title='📈 고객 그룹별 상품 다양성 비교',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_diversity.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_diversity, use_container_width=True)
        st.markdown("""
        **실행 방안**:
        * **신상품 체험 쿠폰**: 2~3등급 고객에게 신상품 또는 특정 카테고리 상품에 대한 할인 쿠폰을 제공합니다.
        * **번들(Bundle) 추천**: 기존 구매 상품과 시너지를 낼 수 있는 다른 상품을 묶어서 추천하고 할인합니다.
        * **개인화된 탐색 추천**: 고객의 과거 구매 이력을 바탕으로, 아직 구매하지 않은 관련 상품을 추천합니다.
        """)

    st.markdown("---")

    with st.expander("2️⃣ **재방문 및 재구매율 증가 전략**"):
        st.markdown("""
        **문제 인식**: 1등급 고객의 **재구매율**이 2~3등급 고객보다 현저히 높습니다.
        **전략 목표**: 2~3등급 고객의 InstaCart 방문 및 구매 빈도를 늘려 재구매율을 개선합니다.
        """)
        fig_reorder = px.bar(compare_df.groupby('group')['reorder_rate'].mean().reset_index(),
                     x='reorder_rate', y='group', orientation='h', color='group',
                     labels={'reorder_rate':'평균 재구매율', 'group':'고객 그룹'},
                     title='📊 고객 그룹별 재구매율 비교',
                     color_discrete_sequence=px.colors.qualitative.D3)
        fig_reorder.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_reorder, use_container_width=True)
        st.markdown("""
        **실행 방안**:
        * **개인화된 리마인더**: 장바구니에 담긴 상품 또는 자주 구매하던 상품의 재고 알림을 보냅니다.
        * **할인 쿠폰 제공**: 특정 기간 내 재구매 시 사용할 수 있는 할인 쿠폰을 제공하여 즉각적인 구매를 유도합니다.
        * **푸시 알림 강화**: 고객이 앱을 잊지 않도록 개인화된 추천 상품 또는 혜택 알림을 보냅니다.
        """)

    st.markdown("---")

    with st.expander("3️⃣ **휴면 방지 및 활동 주기 단축 전략**"):
        st.markdown("""
        **문제 인식**: 2~3등급 고객의 **최근 방문 주기(Recency)**가 1등급 고객보다 길어 휴면 고객으로 전환될 위험이 있습니다.
        **전략 목표**: 고객의 방문 주기를 단축시키고, 비활동 고객으로 전환되는 것을 방지합니다.
        """)
        fig_recency = px.bar(compare_df.groupby('group')['recency'].mean().reset_index(),
                     x='recency', y='group', orientation='h', color='group',
                     labels={'recency':'평균 최근 방문 주기(일)', 'group':'고객 그룹'},
                     title='🕒 고객 그룹별 최근 방문 주기 비교',
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig_recency.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_recency, use_container_width=True)
        st.markdown("""
        **실행 방안**:
        * **장바구니 상기 알림**: 장바구니에 담아둔 상품이 있는 경우, 구매를 독려하는 알림을 보냅니다.
        * **인기 상품/특가 추천**: 고객이 관심을 가질 만한 인기 상품 또는 한정 특가 상품을 추천하여 재방문을 유도합니다.
        * **가입 감사/웰컴 백 쿠폰**: 오랜만에 방문한 고객에게 감사 메시지와 함께 작은 혜택을 제공합니다.
        """)

    st.markdown("---")

    with st.expander("4️⃣ **평균 장바구니 크기 증가 전략**"):
        st.markdown("""
        **문제 인식**: 1등급 고객의 **평균 장바구니 크기**가 2~3등급 고객보다 큽니다.
        **전략 목표**: 2~3등급 고객이 한 번 주문할 때 더 많은 상품을 구매하도록 유도하여 객단가를 높입니다.
        """)
        fig_cart_size = px.bar(compare_df.groupby('group')['avg_cart_size'].mean().reset_index(),
                     x='avg_cart_size', y='group', orientation='h', color='group',
                     labels={'avg_cart_size':'평균 장바구니 크기', 'group':'고객 그룹'},
                     title='🛍️ 고객 그룹별 평균 장바구니 크기 비교',
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_cart_size.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_cart_size, use_container_width=True)
        st.markdown("""
        **실행 방안**:
        * **금액대별 할인/사은품**: 일정 금액 이상 구매 시 할인 또는 사은품을 제공하여 추가 구매를 유도합니다.
        * **'함께 구매하면 좋은 상품' 추천**: 장바구니에 담긴 상품과 연관성이 높은 상품을 추천하여 교차 판매를 증진시킵니다.
        * **무료 배송 임계값 설정**: 무료 배송을 위한 최소 구매 금액을 설정하여 고객이 더 많은 상품을 담도록 합니다.
        """)

    st.markdown("---")

    with st.expander("5️⃣ **재구매 주기 최적화 마케팅 전략**"):
        st.markdown("""
        **문제 인식**: 1등급 고객은 더 짧고 일정한 재구매 주기를 보이는 반면, 2~3등급 고객은 주기가 길고 불규칙합니다.
        **전략 목표**: 2~3등급 고객의 재구매 주기를 1등급 고객과 유사하게 짧고 일관되게 만듭니다.
        """)

        diamond_users = vip_df[vip_df['vip_grade'] == '1.Diamond']['user_id']
        diamond_orders = orders[orders['user_id'].isin(diamond_users)].copy()
        diamond_orders['days_since_prior_order'] = diamond_orders['days_since_prior_order'].fillna(0)
        diamond_avg_interval = diamond_orders.groupby('user_id')['days_since_prior_order'].mean().rename('avg_reorder_interval')

        mid_users = vip_df[vip_df['vip_grade'].isin(['2.Platinum', '3.Gold'])]['user_id']
        mid_orders = orders[orders['user_id'].isin(mid_users)].copy()
        mid_orders['days_since_prior_order'] = mid_orders['days_since_prior_order'].fillna(0)
        mid_avg_interval = mid_orders.groupby('user_id')['days_since_prior_order'].mean().rename('avg_reorder_interval')

        # KDE Plot을 Plotly로 변경 (px.density_kde -> px.histogram + histnorm='density')
        df_interval = pd.DataFrame({
            'avg_reorder_interval': pd.concat([diamond_avg_interval, mid_avg_interval]),
            'group': ['1등급 (Diamond)'] * len(diamond_avg_interval) + ['2~3등급 (Platinum+Gold)'] * len(mid_avg_interval)
        })

        fig_kde = px.histogram(df_interval, x='avg_reorder_interval', color='group',
                                 marginal='box', # 히스토그램 위에 박스 플롯 추가 (선택 사항: 'violin' 또는 'rug')
                                 nbins=30, # 빈 개수
                                 histnorm='density', # 밀도 기준으로 정규화
                                 opacity=0.7, # 투명도 조절
                                 title='🔄 재구매 주기 분포 비교 (평균 재구매 주기)',
                                 labels={'avg_reorder_interval': '재구매 주기 (일)', 'count': '밀도'},
                                 color_discrete_sequence=['#4682B4', '#DAA520'], # SteelBlue, Goldenrod
                                 template='plotly_white'
                                )
        fig_kde.update_layout(bargap=0.1) # 바 간격 설정
        st.plotly_chart(fig_kde, use_container_width=True)

        st.markdown(f"**1등급 고객 평균 재구매 주기:** `{diamond_avg_interval.mean():.2f}`일")
        st.markdown(f"**2~3등급 고객 평균 재구매 주기:** `{mid_avg_interval.mean():.2f}`일")
        st.markdown(f"**1등급 고객 중앙값 재구매 주기:** `{diamond_avg_interval.median():.2f}`일")
        st.markdown(f"**2~3등급 고객 중앙값 재구매 주기:** `{mid_avg_interval.median():.2f}`일")
        st.markdown("""
        **핵심 인사이트**: 1등급 고객의 재구매 주기가 2~3등급 고객보다 짧고, 분포가 특정 일자에 집중되어 있습니다.
        **실행 방안**:
        * **주기 맞춤형 리마케팅**: 2~3등급 고객의 과거 구매 주기를 분석하여, 다음 구매 시점이 도래하기 직전에 맞춤형 상품 추천 또는 할인 메시지를 발송합니다.
        * **정기 배송 서비스 홍보**: 자주 구매하는 상품에 대해 정기 배송 서비스 가입을 유도하여 구매 주기를 고정시키고 편의성을 제공합니다.
        """)


# --- 🎯 맞춤형 추천 시스템 탭 (새로운 UI/UX 적용 - 장바구니 기능 포함) ---
with 탭_추천:
    st.header("🎯 맞춤형 추천 시스템 (ALS 모델 기반)")
    st.markdown("""
    ALS(Alternating Least Squares) 협업 필터링 모델을 사용하여
    **선택된 고객에게 개인화된 맞춤 상품을 추천**합니다.
    이를 통해 고객 만족도를 높이고 구매를 유도할 수 있습니다.
    """)

    st.markdown("---")

    # 고객 선택 섹션 (상단에 배치하여 user_choice가 먼저 정의되도록 함)
    col_selector, col_spacer = st.columns([1, 2])
    with col_selector:
        grade_option = st.selectbox(
            "🔎 **추천받을 고객의 등급을 선택하세요.**",
            options=[
                "1등급 (Diamond)",
                "2~3등급 (Platinum + Gold)"
            ],
            help="1등급은 최상위 고객, 2~3등급은 중상위 고객군을 묶어서 추천 대상을 선택합니다."
        )

        if grade_option == "1등급 (Diamond)":
            selected_grade = '1.Diamond'
        else:
            selected_grade = ['2.Platinum', '3.Gold']

        candidate_users_all = vip_df[vip_df['vip_grade'].isin([selected_grade]) if isinstance(selected_grade, list) else vip_df['vip_grade'] == selected_grade]['user_id'].tolist()
        candidate_users = [user_id for user_id in candidate_users_all if user_id in user_to_idx]

        # candidate_users가 비어있을 경우 처리
        if not candidate_users:
            st.warning("⚠️ 선택된 등급에 해당하는 고객 중 추천 모델에 학습된 고객이 없습니다. 다른 등급을 선택하거나 데이터를 확인해주세요.")
            # 오류를 피하기 위해 user_choice를 None으로 설정하거나, 빈 리스트의 첫 요소를 사용
            user_choice = None # user_choice를 None으로 초기화
        else:
            user_choice = st.selectbox("👤 **추천받을 고객 ID를 선택하세요.**", candidate_users, key="user_select")

    st.markdown("---")

    # user_choice가 선택되었을 때만 하위 정보들을 표시
    if user_choice:
        user_data = vip_df[vip_df['user_id'] == user_choice].iloc[0]

        # 상단 사용자 정보 및 포인트/쿠폰 영역 (이미지 참조)
        st.markdown("##### 쇼핑하기 좋은 날이에요! 😊")
        col_user_info, col_points, col_coupons, col_vouchers = st.columns([0.8, 1, 1, 1])

        with col_user_info:
            st.markdown(f"### 🙋‍♀️ **{user_choice}님** 〉")
            st.markdown(f"###### 💚 {user_data['vip_grade'].split('.')[1]} 등급이네요! (?)")
        with col_points:
            st.metric("✨ L.POINT 〉", f"{np.random.randint(500, 5000):,}P")
        with col_coupons:
            st.metric("🎫 나의쿠폰 〉", f"{np.random.randint(0, 5):,}개")
        with col_vouchers:
            st.metric("🎁 모바일상품권 〉", f"{np.random.randint(0, 10) * 1000:,}원")

        st.markdown("---")

        # 알림/광고 배너
        st.info("🔔 알림 [리서치패널] 참여 시 L.POINT 최대 5만점 적립 〉")
        st.markdown("---")


        # 고객 주요 지표
        st.subheader(f"📋 고객 {user_choice}님의 주요 쇼핑 지표")
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("총 주문 수", f"{int(user_data['total_orders']):,}회")
        with col_metric2:
            st.metric("총 구매 상품 수", f"{int(user_data['total_products']):,}개")
        with col_metric3:
            st.metric("재구매율", f"{user_data['reorder_rate']:.1%}") # 백분율 포맷

        st.markdown("---")

        # 장바구니 현황 표시
        st.subheader("🛒 나의 장바구니")
        if not st.session_state.cart:
            st.info("장바구니가 비어 있습니다. 아래 추천 상품을 담아보세요!")
        else:
            cart_total_price = sum(item['price'] for item in st.session_state.cart)
            st.write(f"현재 장바구니에 **{len(st.session_state.cart)}개**의 상품이 담겨 있으며, 총 예상 금액은 **{int(cart_total_price):,}원**입니다.")
            # 장바구니 목록을 Expander로 보여주기
            with st.expander("장바구니 상품 목록 보기"):
                for i, item in enumerate(st.session_state.cart):
                    st.write(f"{i+1}. {item['name']} - {int(item['price']):,}원")
            if st.button("장바구니 비우기", key="clear_cart"):
                st.session_state.cart = []
                st.rerun() # 장바구니 비운 후 대시보드 새로고침

        st.markdown("---")

        if st.button("✨ **고객 맞춤 추천 상품 보기**", help="선택된 고객에게 개인화된 상품을 추천합니다."):
            with st.spinner(f"🚀 고객 {user_choice}님에게 추천 상품을 계산 중입니다... 잠시만 기다려주세요!"):
                recommended_products_list_info = recommend_products_als(
                    _user_id = user_choice,
                    _model = model_als,
                    _user_to_idx = user_to_idx,
                    _idx_to_product = idx_to_product,
                    _user_item_matrix = user_item_matrix,
                    _products_df = products,
                    N=5
                )

                st.markdown("### 🛍️ 고객님을 위한 추천 상품") # 이미지 속 "고객님을 위한 상품" 헤더
                if recommended_products_list_info:
                    # 3개의 컬럼으로 상품 카드 배치 (이미지와 유사하게)
                    num_product_cols = 3
                    # recommended_products_list_info가 딕셔너리 리스트이므로, 슬라이싱
                    displayed_products = recommended_products_list_info[:5] # 최대 5개만 표시

                    cols = st.columns(num_product_cols)
                    for i, product_info in enumerate(displayed_products):
                        with cols[i % num_product_cols]:
                            # 각 상품 카드 디자인
                            # 상품명은 markdown으로 bold 처리
                            st.markdown(f"**{product_info['product_name']}**")
                            # 이미지 자리 표시 (실제 이미지 파일 없음)
                            st.markdown(f"<div style='border: 1px solid #ddd; padding: 20px; text-align: center; height: 150px; display: flex; align-items: center; justify-content: center; background-color: #f9f9f9;'>상품 이미지</div>", unsafe_allow_html=True)

                            # 할인율 및 원래 가격
                            st.markdown(f"<span style='color: red; font-weight: bold;'>{product_info['discount_rate']}%</span> "
                                        f"<span style='text-decoration: line-through; color: gray; font-size:0.9em;'>{int(product_info['original_price']):,}원</span>",
                                        unsafe_allow_html=True)
                            # 현재 가격
                            st.markdown(f"<span style='font-weight: bold; font-size: 1.2em;'>{int(product_info['current_price']):,}원</span>",
                                        unsafe_allow_html=True)
                            # 리뷰 정보
                            st.markdown(f"⭐ {product_info['review_score']:.1f} 리뷰 {int(product_info['review_count']):,}")

                            # 장바구니 담기 버튼 추가
                            if st.button("🛒 장바구니 담기", key=f"add_cart_{product_info['product_id']}",
                                         on_click=add_to_cart, args=(product_info['product_name'], product_info['current_price'])):
                                pass # on_click 콜백 함수가 이미 처리하므로 여기서는 추가 작업 없음

                            st.markdown("---") # 상품 카드 구분선
                else:
                    st.warning("해당 고객에게 추천 가능한 새로운 상품이 없거나, 이미 모든 상품을 구매했습니다. 다른 고객을 선택해보세요.")
    else:
        st.info("⬆️ 고객 ID를 선택하시면 해당 고객의 정보와 맞춤형 추천 상품을 볼 수 있습니다.") # 고객 선택 전 메시지

st.success("✅ 대시보드 로드 완료")