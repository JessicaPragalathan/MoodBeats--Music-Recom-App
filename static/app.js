document.addEventListener('DOMContentLoaded', () => {
    const formOpenBtn = document.getElementById('form-open');
    const home = document.querySelector('.home');
    const formContainer = document.querySelector('.form_container');
    const formCloseBtn = document.querySelector('.form_close');
    const signupBtn = document.getElementById('signup');
    const loginBtn = document.getElementById('login');
    const loginForm = document.querySelector('.login_form');
    const signupForm = document.querySelector('.signup_form');

    formOpenBtn.addEventListener('click', () => formContainer.classList.add('active'));
    formCloseBtn.addEventListener('click', () => formContainer.classList.remove('active'));

    signupBtn.addEventListener('click', (e) => {
        e.preventDefault();
        loginForm.style.display = 'none';
        signupForm.style.display = 'block';
    });

    loginBtn.addEventListener('click', (e) => {
        e.preventDefault();
        signupForm.style.display = 'none';
        loginForm.style.display = 'block';
    });
});

// document.addEventListener('DOMContentLoaded', () => {
//     const formCloseBtn = document.querySelector('.form_close');
//     const signupBtn = document.querySelector('.login_signup a');

//     formCloseBtn.addEventListener('click', () => {
//         document.querySelector('.form_container').classList.remove('active');
//     });

//     signupBtn.addEventListener('click', (e) => {
//         e.preventDefault();
//         document.querySelector('.login_form').style.display = 'none';
//         document.querySelector('.signup_form').style.display = 'block';
//     });

//     document.querySelector('.login_signup a').addEventListener('click', (e) => {
//         e.preventDefault();
//         document.querySelector('.signup_form').style.display = 'none';
//         document.querySelector('.login_form').style.display = 'block';
//     });
// });

