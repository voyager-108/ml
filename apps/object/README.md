<div align="center" class="centered">

# :bucket: Хранилище 

</div>

В качестве хранилища промежуточных объектов мы используем сервис Yandex.Cloud Object Storage. Подробная информация о сервисе доступна в [документации](https://cloud.yandex.ru/docs/storage/). Для использования вашего бакета вместо нашего, необходимо передать его название в переменную окружения `OBJECT_STORE_BUCKET`.

```bash

export OBJECT_STORE_BUCKET=<your-bucket-name>

```

Для корректной работы, вы должны либо авторизоваться, либо предоставить всем пользователям возможность исполнения PUT и GET в ваш бакет (назначить роль `storage.editor` группе `allUsers`).
