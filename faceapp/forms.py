# from django import forms
# from .models import User
#
#
# class UserRegistrationForm(forms.ModelForm):
#     # password = forms.CharField(widget=forms.PasswordInput)
#     # password_confirm = forms.CharField(widget=forms.PasswordInput)
#     photo = forms.CharField(widget=forms.HiddenInput())
#
#     class Meta:
#         model = User
#         fields = ['login', 'name_user', 'biograghy', 'my_skills',  'photo']
#         # fields = ['login', 'name', 'biograghy', 'my_skills', 'path_to_folder', 'path_to_faces', 'password', 'password_confirm', 'photo']
#
#     def clean(self):
#         cleaned_data = super().clean()
#         # password = cleaned_data.get('password')
#         # password_confirm = cleaned_data.get('password_confirm')
#         #
#         # if password != password_confirm:
#         #     raise forms.ValidationError("Passwords do not match")
#
#         if not cleaned_data.get('photo'):
#             raise forms.ValidationError("Photo is required")
#
#         return cleaned_data
