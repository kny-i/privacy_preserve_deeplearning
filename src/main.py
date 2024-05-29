import numpy as np

class ParameterServer:
    def __init__(self, num_params):
        self.w_global = np.random.randn(num_params)  # グローバルパラメータの初期化
        self.stat = np.zeros(num_params)  # statベクトルの初期化

    def upload_gradients(self, gradients, indices):
        for i, grad in zip(indices, gradients):
            self.w_global[i] += grad
            self.stat[i] += 1

    def download_parameters(self, theta_d):
        num_params_to_download = int(theta_d * len(self.w_global))
        sorted_indices = np.argsort(-self.stat)  # statを降順にソート
        indices_to_download = sorted_indices[:num_params_to_download]
        return self.w_global[indices_to_download], indices_to_download

class Client:
    def __init__(self, server, client_id, num_params, learning_rate=0.01):
        self.server = server
        self.client_id = client_id
        self.w_local = np.random.randn(num_params)  # ローカルパラメータの初期化
        self.learning_rate = learning_rate

    def local_sgd(self, data):
        # データを使ってローカルSGDを実行（ここでは簡易的にランダムな勾配を使用）
        gradients = np.random.randn(len(self.w_local))
        self.w_local -= self.learning_rate * gradients
        return gradients

    def upload_gradients(self, gradients, theta_u):
        num_params_to_upload = int(theta_u * len(self.w_local))
        sorted_indices = np.argsort(-np.abs(gradients))  # 勾配を絶対値の降順にソート
        indices_to_upload = sorted_indices[:num_params_to_upload]
        self.server.upload_gradients(gradients[indices_to_upload], indices_to_upload)

    def download_parameters(self, theta_d):
        new_params, indices = self.server.download_parameters(theta_d)
        before_update = self.w_local.copy()  # ダウンロード前のローカルパラメータをコピー
        self.w_local[indices] = new_params
        return before_update, self.w_local, indices  # ダウンロード前後のローカルパラメータを返す

if __name__ == "__main__":
    # サーバーの初期化
    num_params = 100
    server = ParameterServer(num_params)

    # クライアントの初期化
    client_id = 1
    client = Client(server, client_id, num_params)

    # ローカルデータセットでSGDを実行
    data = None  # データセット（ここでは未使用）
    gradients = client.local_sgd(data)
    print("Initial local parameters:", client.w_local)

    # サーバーに勾配をアップロード
    theta_u = 0.2
    client.upload_gradients(gradients, theta_u)

    # サーバーからパラメータをダウンロード
    theta_d = 0.2
    before_update, after_update, updated_indices = client.download_parameters(theta_d)

    # 結果を表示
    print("Updated local parameters:", after_update)
    print("Global parameters on server:", server.w_global)

    # ダウンロード前後の差分を表示
    print("Indices of updated parameters:", updated_indices)
    print("Parameters before update:", before_update[updated_indices])
    print("Parameters after update:", after_update[updated_indices])
    print("Difference:", after_update[updated_indices] - before_update[updated_indices])
