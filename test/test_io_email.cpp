#include <io/IBTimer.h>
#include <io/smtp.h>

int test_with_qq_email() {
    IBTimer timer("Send an email through QQ SMTP.");

    std::string to      = "mapengfei@mail.nwpu.edu.cn";
    std::string content = "成功发送邮件！";
    smtp::CSmtp email_qq(25,                 /*smtp端口*/
                         "smtp.qq.com",      /*smtp服务器地址*/
                         "499908174@qq.com", /*你的邮箱地址*/
                         "kxhxvwrymfppbicc", /*邮箱密码*/
                         to,                 /*目的邮箱地址*/
                         "测试邮箱",         /*主题*/
                         content             /*邮件正文*/
    );

    return smtp::email_notification(email_qq) == 0;
}

int test_with_163_email() {
    IBTimer timer("Send an email through 163 SMTP.");
    // std::string to = "mapengfei@mail.nwpu.edu.cn";
    std::string to      = "2968721968@qq.com";
    std::string content = "成功发送邮件！";
    smtp::CSmtp email_163(25,                 /*smtp端口*/
                          "smtp.163.com",     /*smtp服务器地址*/
                          "nwpumpf@163.com",  /*你的邮箱地址*/
                          "JVEKPVNLIPWFMASA", /*邮箱密码*/
                          to,                 /*目的邮箱地址*/
                          "测试邮箱",         /*主题*/
                          content             /*邮件正文*/
    );

    return smtp::email_notification(email_163) == 0;
}

int main() {
    test_with_qq_email();
    test_with_163_email();
}
