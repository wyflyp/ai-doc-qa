-- 在 MySQL / Navicat / MySQL Workbench 里手动执行这一句就好
-- 表结构由 db.py 的 init_db() 自动创建，不用在这里建表

CREATE DATABASE IF NOT EXISTS ai_doc_platform
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- 注意：数据库名要和 .env 里的 DB_NAME 完全一致

-- 如果密码忘了，Windows 管理员命令行执行：
--   mysql -u root -p
-- 然后：
--   ALTER USER 'root'@'localhost' IDENTIFIED BY '你的新密码';
--   FLUSH PRIVILEGES;
-- 还是不行就重置：
--   mysqld --skip-grant-tables --console
-- 另开一个窗口进入 mysql 后：
--   UPDATE mysql.user SET authentication_string=PASSWORD('新密码') WHERE User='root';
--   FLUSH PRIVILEGES;